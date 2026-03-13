import torch
from .base_scorer import BaseScorer
import json
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from .utils import get_total_lines
from tqdm import tqdm
from utils.utils_jsonl import append_jsonl, repair_trailing_incomplete_jsonl, load_jsonl_id_set, normalize_id


class DeitaQScorer(BaseScorer):
    def _validate_config(self):
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model.")
            self.config['model'] = 'hkust-nlp/deita-quality-scorer'


        if ("max_length" in self.config and isinstance(self.config["max_length"], int)
                and 0 < self.config["max_length"] <= 2048):
            print(f"Using specified max_length: {self.config['max_length']}.")
        elif "max_length" in self.config and isinstance(self.config["max_length"], int) and self.config["max_length"] <= 0:
            print(
                "Warning: the specific max_length should > 0. use default value of 2048.")
            self.config['max_length'] = 2048
        elif "max_length" in self.config and isinstance(self.config["max_length"], int) and self.config["max_length"] > 2048:
            print(
                "Warning: the specific max_length should not be larger than 2048. use default value of 2048.")
            self.config['max_length'] = 2048
        else:
            print("Warning: No specific max_length, use default value of 2048.")
            self.config['max_length'] = 2048

        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 32
            print("Warning: No/invalid batch_size, use default value of 32.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

    def _setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['model'])
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model'])
        except Exception as e:
            print(
                f"Warnning: Specified Model Path Does not Work ({e}), Use Remote Model Instead.")
            self.model = AutoModelForCausalLM.from_pretrained(
                'hkust-nlp/deita-quality-scorer')
            self.tokenizer = AutoTokenizer.from_pretrained(
                'hkust-nlp/deita-quality-scorer')

        self.model.to(self.device)
        self.model.eval()
        print("Setting up DeitaQScorer successfully")

        self.id2score = {
            29896: "1",
            29906: "2",
            29941: "3",
            29946: "4",
            29945: "5",
            29953: "6"
        }
        self._score_ids = torch.tensor(
            list(self.id2score.keys()), device=self.device, dtype=torch.long)
        self._score_values = torch.tensor(
            [1, 2, 3, 4, 5, 6], device=self.device, dtype=torch.float)

    def score_item(self, data_item):

        score = self.score_batch([data_item])[0]
        return score

    def score_batch(self, data_items: List[Dict]) -> List[float]:
        quality_template = (
            "You are a helpful assistant. Please identify the quality score of the Response corresponding to the Question. \n"
            "#Question#:\n{instruction}\n#Response#:\n{output} \n##Quality: "
        )

        valid_indices = []
        prompts = []
        for idx, item in enumerate(data_items):
            instr = item["instruction"].strip()
            input_text = item.get("input", "").strip()
            outp = item["output"].strip()
            if input_text:
                instr = instr + "\n" + input_text

            prompts.append(quality_template.format(
                instruction=instr, output=outp))
            valid_indices.append(idx)

        scores = [None] * len(data_items)
        if len(valid_indices) == 0:
            return scores

        # Check for sequences exceeding max_length
        valid_prompts = [prompts[i] for i in valid_indices]
        # valid_prompts = [prompts[i][:10240] for i in valid_indices]
        tokenized_no_trunc = self.tokenizer(
            valid_prompts,
            padding=False,
            truncation=False,
            return_tensors=None
        )
        max_length = self.config["max_length"]
        for idx, input_ids in enumerate(tokenized_no_trunc["input_ids"]):
            if len(input_ids) > max_length:
                original_idx = valid_indices[idx]
                print(f"Warning: Data item at index {original_idx} has length {len(input_ids)} which exceeds max_length {max_length}. It will be truncated.")

        enc = self.tokenizer(
            valid_prompts,
            padding=True,
            truncation=True,
            max_length=self.config["max_length"],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **enc,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True
            )

            first_token_scores = outputs.scores[0]  # [B, V]

            try:
                selected_logits = first_token_scores.index_select(
                    1, self._score_ids)  # [B, 6]
            except Exception:

                for i in valid_indices:
                    scores[i] = 3.0
                return scores

            probs = torch.softmax(selected_logits, dim=-1)         # [B, 6]
            batch_scores = (probs * self._score_values).sum(-1)    # [B]

        for j, idx in enumerate(valid_indices):
            scores[idx] = batch_scores[j].item()

        return scores

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config["batch_size"]
        buf_items, buf_ids = [], []

        with open(dataset, 'r') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get(
                'name', 'DeitaQScorer'))
            for line in f:
                item = json.loads(line.strip())
                buf_items.append(item)
                buf_ids.append(item.get("id", ""))

                if len(buf_items) == batch_size:
                    batch_scores = self.score_batch(buf_items)
                    results.extend(
                        {"id": _id, "score": sc}
                        for _id, sc in zip(buf_ids, batch_scores)
                    )
                    buf_items.clear()
                    buf_ids.clear()
                pbar.update(1)

            if buf_items:
                batch_scores = self.score_batch(buf_items)
                results.extend(
                    {"id": _id, "score": sc}
                    for _id, sc in zip(buf_ids, batch_scores)
                )
                buf_items.clear()
                buf_ids.clear()
            pbar.close()

        return results

    def evaluate_to_file(self, dataset: str, output_file: str, resume: bool = True) -> str:
        """
        Stream pointwise results to JSONL (append), enabling resume from existing output_file.
        Assumption: one output JSON line per input JSON line, and order is consistent.
        """
        num_lines = get_total_lines(dataset)
        batch_size = self.config.get("batch_size")

        done_ids = set()
        if resume and os.path.exists(output_file):
            repair_trailing_incomplete_jsonl(output_file)
            done_ids = load_jsonl_id_set(output_file, id_key="id")
            if done_ids:
                print(f"[DeitaQScorer] Resume enabled. Found {len(done_ids)} unique completed ids in {output_file}.")

        if not resume:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8"):
                pass

        buf_items, buf_ids = [], []

        with open(dataset, "r", encoding="utf-8", errors="ignore") as f:
            pbar = tqdm(total=num_lines, desc=self.config.get("name", "DeitaQScorer"))

            for line in f:
                line = line.strip()
                if not line:
                    pbar.update(1)
                    continue
                item = json.loads(line)
                item_id = normalize_id(item.get("id", ""))
                if item_id and item_id in done_ids:
                    pbar.update(1)
                    continue

                buf_items.append(item)
                buf_ids.append(item.get("id", ""))

                if len(buf_items) == batch_size:
                    batch_scores = self.score_batch(buf_items)
                    records = [{"id": _id, "score": sc} for _id, sc in zip(buf_ids, batch_scores)]
                    append_jsonl(records, output_file, flush=True)
                    for _id in buf_ids:
                        nid = normalize_id(_id)
                        if nid:
                            done_ids.add(nid)
                    buf_items.clear()
                    buf_ids.clear()
                pbar.update(1)

            if buf_items:
                batch_scores = self.score_batch(buf_items)
                records = [{"id": _id, "score": sc} for _id, sc in zip(buf_ids, batch_scores)]
                append_jsonl(records, output_file, flush=True)
                for _id in buf_ids:
                    nid = normalize_id(_id)
                    if nid:
                        done_ids.add(nid)
                buf_items.clear()
                buf_ids.clear()

            pbar.close()

        return output_file
