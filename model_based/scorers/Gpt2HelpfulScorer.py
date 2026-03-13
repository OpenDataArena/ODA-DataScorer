import os
import json
from typing import Dict, List, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from .base_scorer import BaseScorer
from .utils import get_total_lines
from utils.utils_jsonl import (
    append_jsonl,
    load_jsonl_id_set,
    normalize_id,
    repair_trailing_incomplete_jsonl,
)


class Gpt2HelpfulScorer(BaseScorer):
    """
    GPT2-based helpfulness reward model scorer
    using Ray2333/gpt2-large-helpful-reward_model.
    """

    DEFAULT_MODEL = "Ray2333/gpt2-large-helpful-reward_model"

    # ====================== Config Validation ======================
    def _validate_config(self):
        # 1. Model path
        model = self.config.get("model", None)
        if not model:
            print(f"Warning: no model specified, using default {self.DEFAULT_MODEL}")
            self.config["model"] = self.DEFAULT_MODEL


        # 2. batch_size
        if (
            "batch_size" not in self.config
            or not isinstance(self.config["batch_size"], int)
            or self.config["batch_size"] <= 0
        ):
            self.config["batch_size"] = 8
            print("Warning: invalid batch_size, using default=8.")

        # 3. max_length
        if (
            "max_length" not in self.config
            or not isinstance(self.config["max_length"], int)
            or self.config["max_length"] <= 0
            or self.config["max_length"] > 1024
        ):
            self.config["max_length"] = 1024
            print("Warning: invalid max_length, using default=1024.")
        else:
            print(f"Using specified max_length={self.config['max_length']}.")


    # ====================== Model Setup ======================
    def _setup(self):
        model_name_or_path = self.config["model"]

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=1,
            torch_dtype=torch.bfloat16
        )

        # Set pad_token_id to avoid errors when batch > 1
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

        print("✅ Gpt2HelpfulScorer setup complete.")

    # ====================== Core Logic ======================
    @staticmethod
    def _build_qa(item: Dict) -> Tuple[str, str]:
        instruction = item["instruction"]
        input_text = item.get("input", "")
        response = item["output"]

        if input_text:
            q = f"\n\nHuman: {instruction}\n{input_text}\n\nAssistant:"
        else:
            q = f"\n\nHuman: {instruction}\n\nAssistant:"
        a = response
        return q, a

    def score_item(self, data_item: Dict) -> float:
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[float]:
        """
        Batch inference with max_length limitation.
        """
        if not data_items:
            return []

        questions, answers = [], []
        for item in data_items:
            q, a = self._build_qa(item)
            questions.append(q)
            answers.append(a)

        # Check if any text exceeds max_length before tokenization
        for idx, (q, a) in enumerate(zip(questions, answers)):
            combined_text = q + a
            # Rough estimation: 1 token ≈ 4 characters
            estimated_tokens = len(combined_text) // 4
            if estimated_tokens > self.config["max_length"]:
                item_id = data_items[idx].get("id", idx)
                print(f"Warning: Item {item_id} estimated length ({estimated_tokens} tokens) exceeds max_length ({self.config['max_length']}), will be truncated.")

        # Key point: max_length is used in tokenizer
        inputs = self.tokenizer(
            questions,
            answers,
            return_tensors="pt",
            truncation=True,
            padding=True,                     # Pad to max_length
            max_length=self.config["max_length"],     # Control maximum length
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Model forward pass, no need to pass max_length
        with torch.no_grad():
            logits = self.model(**inputs).logits  # shape [B, 1]
            rewards = logits.squeeze(-1).cpu().float().tolist()

        # Ensure list[float] format
        if isinstance(rewards, float):
            rewards = [rewards]

        return [float(r) for r in rewards]

    # ====================== Evaluation Logic ======================
    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config["batch_size"]
        buffer_items, buffer_ids = [], []

        with open(dataset, "r", encoding="utf-8") as f:
            pbar = tqdm(total=num_lines, desc="Gpt2HelpfulScorer")
            for line in f:
                item = json.loads(line.strip())
                buffer_items.append(item)
                buffer_ids.append(item.get("id", ""))

                if len(buffer_items) == batch_size:
                    scores = self.score_batch(buffer_items)
                    results.extend(
                        {"id": i, "score": s}
                        for i, s in zip(buffer_ids, scores)
                    )
                    buffer_items.clear()
                    buffer_ids.clear()
                pbar.update(1)

            if buffer_items:
                scores = self.score_batch(buffer_items)
                results.extend(
                    {"id": i, "score": s}
                    for i, s in zip(buffer_ids, scores)
                )
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
                    f"[Gpt2HelpfulScorer] Resume enabled. Found {len(done_ids)} unique completed ids in {output_file}."
                )

        if not resume:
            dir_path = os.path.dirname(output_file)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(output_file, "w", encoding="utf-8"):
                pass

        buf_items, buf_ids = [], []

        with open(dataset, "r", encoding="utf-8", errors="ignore") as f:
            pbar = tqdm(total=num_lines, desc=self.config.get("name", "Gpt2HelpfulScorer"))

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