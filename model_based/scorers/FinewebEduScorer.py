import json
import os
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .base_scorer import BaseScorer
from .utils import get_total_lines
from utils.utils_jsonl import (
    append_jsonl,
    load_jsonl_id_set,
    normalize_id,
    repair_trailing_incomplete_jsonl,
)


class FinewebEduScorer(BaseScorer):
    def _validate_config(self):
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model.")
            self.config['model'] = 'HuggingFaceFW/fineweb-edu-classifier'
        else:
            if self.config['model'] == 'HuggingFaceFW/fineweb-edu-classifier':
                print("Downloading and use the specific remote huggingface model.")
            elif not os.path.exists(self.config["model"]):
                print(
                    f"Warning: Specified local model path '{self.config['model']}' does not exist. "
                    "Downloading the remote huggingface model: HuggingFaceFW/fineweb-edu-classifier"
                )
                self.config['model'] = 'HuggingFaceFW/fineweb-edu-classifier'
            else:
                print(
                    f"Using specified local model: '{self.config['model']}'. ")

        if "max_length" in self.config and isinstance(self.config["max_length"], int) and 0 < self.config["max_length"] <= 2048:
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
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config['model'])
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model'])
        except Exception as e:
            print(
                f"Load specified model failed ({e}), fall back to HuggingFaceFW/fineweb-edu-classifier")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                'HuggingFaceFW/fineweb-edu-classifier')
            self.tokenizer = AutoTokenizer.from_pretrained(
                'HuggingFaceFW/fineweb-edu-classifier')

        self.model.to(self.device)
        self.model.eval()
        print("Setting up FinewebEduScorer successfully")

    def score_item(self, data_item):
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[float]:
        # Extract text content from data items, concatenating instruction, input, and output
        texts = []
        for item in data_items:
            parts = []
            parts.append(item["instruction"])
            parts.append(item.get("input", ""))
            parts.append(item["output"])

            # Concatenate with newlines
            text = "\n".join(parts) if parts else ""
            texts.append(text)

        # Batch tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.config["max_length"]
        ).to(self.device)

        # Check if any sequences were truncated
        for i, input_ids in enumerate(inputs["input_ids"]):
            if len(input_ids) >= self.config["max_length"]:
                print(f"Warning: Data item at index {i} exceeds max_length ({self.config['max_length']}). Text has been truncated.")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(-1).float()
            scores = logits.cpu().tolist()
            
            # Ensure returning a list even if there's only one sample
            if not isinstance(scores, list):
                scores = [scores]

        return scores

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config.get("batch_size")
        buffer_items = []
        buffer_ids = []

        with open(dataset, 'r', encoding="utf-8", errors="ignore") as f:
            pbar = tqdm(total=num_lines, desc=self.config.get(
                'name', 'FinewebEduScorer'))
            for line in f:
                item = json.loads(line.strip())
                buffer_items.append(item)
                buffer_ids.append(item.get("id", ""))

                if len(buffer_items) == batch_size:
                    batch_scores = self.score_batch(buffer_items)
                    results.extend([
                        {"id": id_, "score": sc}
                        for id_, sc in zip(buffer_ids, batch_scores)
                    ])
                    buffer_items.clear()
                    buffer_ids.clear()
                pbar.update(1)

            if buffer_items:
                batch_scores = self.score_batch(buffer_items)
                results.extend([
                    {"id": id_, "score": sc}
                    for id_, sc in zip(buffer_ids, batch_scores)
                ])
                buffer_items.clear()
                buffer_ids.clear()
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
                    f"[FinewebEduScorer] Resume enabled. Found {len(done_ids)} unique completed ids in {output_file}."
                )

        if not resume:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, "w", encoding="utf-8"):
                pass

        buf_items, buf_ids = [], []

        with open(dataset, "r", encoding="utf-8", errors="ignore") as f:
            pbar = tqdm(total=num_lines, desc=self.config.get("name", "FinewebEduScorer"))

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
