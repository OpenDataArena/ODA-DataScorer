import re
import os
import json
import fasttext
from huggingface_hub import hf_hub_download
from .base_scorer import BaseScorer
from typing import Dict, List
from tqdm import tqdm
from .utils import get_total_lines
from utils.utils_jsonl import (
    append_jsonl,
    load_jsonl_id_set,
    normalize_id,
    repair_trailing_incomplete_jsonl,
)


class TextbookScorer(BaseScorer):
    score_dict = {
        '__label__': 0,
        '__label__Low': 0,
        '__label__Mid': 1,
        '__label__High': 2
    }

    def replace_newlines(self, text: str) -> str:
        return re.sub("\n+", " ", text)

    def _validate_config(self):
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model.")
            self.config['model'] = 'kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2'
        else:
            print(f"Using specified model: '{self.config['model']}'.")

        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 32
            print("Warning: No/invalid batch_size, use default value of 32.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

    def _setup(self):
        try:
            if self.config['model'] == 'kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2':
                self.model = fasttext.load_model(hf_hub_download(
                    self.config['model'], "model.bin"))
            else:
                path = f"{str(self.config['model'])}/model.bin"
                self.model = fasttext.load_model(path)
        except Exception as e:
            print(
                f"Load specified model failed ({e}), fall back to kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2")
            self.model = fasttext.load_model(hf_hub_download(
                'kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2', "model.bin"))
        print("Setting up TextbookScorer successfully")

    def score_item(self, data_item):
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[float]:
        texts = []
        for item in data_items:
            instruction = item["instruction"]
            response = item["output"]
            if "input" in item:
                input_text = item["input"]
                text = instruction + '\n' + input_text + '\n' + response
            else:
                text = instruction + '\n' + response
            text = self.replace_newlines(text)
            texts.append(text)

        labels_batch, probs_batch = self.model.predict(texts, k=-1)
        
        scores = []
        for labels, probs in zip(labels_batch, probs_batch):
            score = 0.0
            for label, prob in zip(labels, probs):
                score += self.__class__.score_dict.get(label, 0) * prob
            scores.append(float(score))

        return scores

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config.get("batch_size")
        buffer_items = []
        buffer_ids = []

        with open(dataset, 'r') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get(
                'name', 'TextbookScorer'))
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
                    f"[TextbookScorer] Resume enabled. Found {len(done_ids)} unique completed ids in {output_file}."
                )

        if not resume:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, "w", encoding="utf-8"):
                pass

        buf_items, buf_ids = [], []

        with open(dataset, "r", encoding="utf-8", errors="ignore") as f:
            pbar = tqdm(total=num_lines, desc=self.config.get("name", "TextbookScorer"))

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
