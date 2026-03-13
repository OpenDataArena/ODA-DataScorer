import torch
import torch.nn as nn
import json
import os
from typing import Dict, List, Optional
from transformers import LlamaPreTrainedModel, LlamaModel, AutoTokenizer, PreTrainedTokenizerFast
from tqdm import tqdm

from .base_scorer import BaseScorer
from .utils import get_total_lines
from utils.utils_jsonl import append_jsonl, repair_trailing_incomplete_jsonl, load_jsonl_id_set, normalize_id


class AtheneForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.CLS_ID = 128003
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )
        hidden_states = transformer_outputs.hidden_states[-1]
        rewards = self.v_head(hidden_states).squeeze(-1)

        bs = int(input_ids.shape[0])
        scores = []
        for i in range(bs):
            c_inds = (input_ids[i] == self.CLS_ID).nonzero(as_tuple=False)
            if len(c_inds) > 0:
                c_ind = c_inds[-1].item()
                scores.append(rewards[i, c_ind])
            else:
                # If no CLS token found, use the last token
                scores.append(rewards[i, -1])
        scores = torch.stack(scores)
        return {"scores": scores}


class AtheneScorer(BaseScorer):
    def _validate_config(self):
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model.")
            self.config['model'] = 'Nexusflow/Athene-RM-8B'

        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 32
            print("Warning: No/invalid batch_size, use default value of 32.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

        if "max_length" not in self.config or not isinstance(self.config["max_length"], int) or self.config["max_length"] <= 0:
            self.config["max_length"] = 4096
            print("Warning: No/invalid max_length, use default value of 4096.")
        else:
            print(f"Using specified max_length: {self.config['max_length']}.")

    def _setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.rank_model = AtheneForSequenceClassification.from_pretrained(
                self.config['model'],
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model'])
        except Exception as e:
            print(
                f"Warning: Specified Model Path Does not Work ({e}), Use Remote Model Instead.")
            self.rank_model = AtheneForSequenceClassification.from_pretrained(
                "Nexusflow/Athene-RM-8B",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Nexusflow/Athene-RM-8B")

        if not torch.cuda.is_available() or self.rank_model.device.type != "cuda":
            self.rank_model.to(self.device)
        self.rank_model.eval()
        print("Setting up AtheneScorer successfully")

    def score_item(self, data_item):
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[float]:

        input_ids_list = []

        for item in data_items:
            prompt = item["instruction"]
            input_text = item.get("input", "")
            if input_text:
                prompt = prompt + "\n" + input_text
            output = item["output"]
            conv = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output}
            ]

            formatted = self.tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Add CLS token
            formatted = formatted + self.tokenizer.cls_token

            encoded = self.tokenizer(
                formatted,
                return_tensors="pt",
                max_length=self.config["max_length"],
                padding=False,
                truncation=True,
            )

            if encoded.input_ids.shape[1] > self.config["max_length"]:
                item_id = item.get("id", "unknown")
                print(f"Warning: Data item (id: {item_id}) exceeds max_length ({self.config['max_length']}), truncating from {encoded.input_ids.shape[1]} tokens.")
                encoded.input_ids = encoded.input_ids[:, -self.config["max_length"]:]
            input_ids_list.append(encoded.input_ids[0])

        batch = self.tokenizer.pad(
            {"input_ids": input_ids_list},
            padding=True,
            return_tensors="pt"
        )
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.rank_model(
                input_ids=input_ids, attention_mask=attention_mask)
            scores = outputs["scores"].float().tolist()

        return [float(s) for s in scores]

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config["batch_size"]
        buf_items, buf_ids = [], []

        with open(dataset, 'r') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get(
                'name', 'AtheneScorer'))
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
        Resume logic is id-based: skip any sample whose id already exists in output_file.
        """
        num_lines = get_total_lines(dataset)
        batch_size = self.config.get("batch_size")

        done_ids = set()
        if resume and os.path.exists(output_file):
            repair_trailing_incomplete_jsonl(output_file)
            done_ids = load_jsonl_id_set(output_file, id_key="id")
            if done_ids:
                print(f"[AtheneScorer] Resume enabled. Found {len(done_ids)} unique completed ids in {output_file}.")

        if not resume:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8"):
                pass

        buf_items, buf_ids = [], []

        with open(dataset, "r", encoding="utf-8", errors="ignore") as f:
            pbar = tqdm(total=num_lines, desc=self.config.get("name", "AtheneScorer"))

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
