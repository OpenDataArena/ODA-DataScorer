import json
import os
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base_scorer import BaseScorer
from .utils import get_total_lines
from utils.utils_jsonl import (
    append_jsonl,
    load_jsonl_id_set,
    normalize_id,
    repair_trailing_incomplete_jsonl,
)


class SkyworkQwenScorer(BaseScorer):
    def _validate_config(self):
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model."
            )
            self.config["model"] = "Skywork/Skywork-Reward-V2-Qwen3-8B"
        else:
            if self.config["model"] == "Skywork/Skywork-Reward-V2-Qwen3-8B":
                print("Downloading and use the specific remote huggingface model.")
            elif not os.path.exists(self.config["model"]):
                print(
                    f"Warning: Specified local model path '{self.config['model']}' does not exist. "
                    "Downloading the remote huggingface model: Skywork/Skywork-Reward-V2-Qwen3-8B"
                )
                self.config["model"] = "Skywork/Skywork-Reward-V2-Qwen3-8B"
            else:
                print(f"Using specified local model: '{self.config['model']}'. ")

        if (
            "batch_size" not in self.config
            or not isinstance(self.config["batch_size"], int)
            or self.config["batch_size"] <= 0
        ):
            self.config["batch_size"] = 32
            print("Warning: No/invalid batch_size, use default value of 32.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

        if (
            "max_length" not in self.config
            or not isinstance(self.config["max_length"], int)
            or self.config["max_length"] <= 0
        ):
            self.config["max_length"] = 4096
            print("Warning: No/invalid max_length, use default value of 4096.")
        else:
            print(f"Using specified max_length: {self.config['max_length']}.")

    def _setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.rank_model = AutoModelForSequenceClassification.from_pretrained(
                self.config["model"],
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
                num_labels=1,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"])
        except Exception as e:
            print(
                f"Warning: Specified model path does not work ({e}), use remote model instead."
            )
            self.rank_model = AutoModelForSequenceClassification.from_pretrained(
                "Skywork/Skywork-Reward-V2-Qwen3-8B",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
                num_labels=1,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Skywork/Skywork-Reward-V2-Qwen3-8B"
            )

        # Ensure padding works for batch scoring
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.rank_model.config, "pad_token_id", None) is None:
            self.rank_model.config.pad_token_id = self.tokenizer.pad_token_id

        self.rank_model.to(self.device)
        self.rank_model.eval()
        print("Setting up SkyworkQwenScorer successfully")

    def score_item(self, data_item):
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[float]:
        """
        Score a batch of data items using a reward-style sequence classification model.
        Each item is expected to have: instruction (prompt) and output (assistant response).
        """
        conversations = []
        for item in data_items:
            prompt = item["instruction"]
            if "input" in item:
                input_text = item["input"]
                prompt = prompt + '\n' + input_text
            output = item["output"]
            conv = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output}
            ]
            conversations.append(conv)

        input_ids_list = []
        truncated_indices = []
        max_length_config = self.config["max_length"]

        for idx, conv in enumerate(conversations):
            encoded = self.tokenizer.apply_chat_template(
                conv,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=False,
                truncation=True,
                max_length=max_length_config,
            )

            actual_length = encoded.shape[1]
            if actual_length >= max_length_config:
                truncated_indices.append(idx)

            input_ids_list.append(encoded[0])

        if truncated_indices:
            item_ids = [data_items[i].get("id", f"index_{i}") for i in truncated_indices]
            print(
                f"Warning: {len(truncated_indices)} item(s) exceeded max_length ({max_length_config}) and were truncated. "
                f"Item IDs: {item_ids[:5]}{'...' if len(item_ids) > 5 else ''}"
            )

        batch = self.tokenizer.pad(
            {"input_ids": input_ids_list},
            padding="longest",
            return_tensors="pt",
            max_length=max_length_config,
        )
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.rank_model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, 1]
            scores = logits.squeeze(-1).float().tolist()

        return [float(s) for s in scores]

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config["batch_size"]
        buf_items, buf_ids = [], []

        with open(dataset, "r", encoding="utf-8", errors="ignore") as f:
            pbar = tqdm(total=num_lines, desc=self.config.get("name", "SkyworkQwenScorer"))
            for line in f:
                item = json.loads(line.strip())
                buf_items.append(item)
                buf_ids.append(item.get("id", ""))

                if len(buf_items) == batch_size:
                    batch_scores = self.score_batch(buf_items)
                    results.extend(
                        {"id": _id, "score": sc} for _id, sc in zip(buf_ids, batch_scores)
                    )
                    buf_items.clear()
                    buf_ids.clear()
                pbar.update(1)

            if buf_items:
                batch_scores = self.score_batch(buf_items)
                results.extend(
                    {"id": _id, "score": sc} for _id, sc in zip(buf_ids, batch_scores)
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
                print(
                    f"[SkyworkQwenScorer] Resume enabled. Found {len(done_ids)} unique completed ids in {output_file}."
                )

        if not resume:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8"):
                pass

        buf_items, buf_ids = [], []

        with open(dataset, "r", encoding="utf-8", errors="ignore") as f:
            pbar = tqdm(total=num_lines, desc=self.config.get("name", "SkyworkQwenScorer"))

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
