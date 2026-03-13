import json
import math
import os
from typing import Dict, List

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .base_scorer import BaseScorer
from .utils import get_total_lines
from utils.utils_jsonl import (
    append_jsonl,
    load_jsonl_id_set,
    normalize_id,
    repair_trailing_incomplete_jsonl,
)


class ThinkingProbScorer(BaseScorer):
    def _validate_config(self):
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model: THU-KEG/AdaptThink-7B-delta0.05.")
            self.config['model'] = 'THU-KEG/AdaptThink-7B-delta0.05'
        else:
            if not os.path.exists(self.config["model"]):
                print(
                    f"Warning: Specified local model path '{self.config['model']}' does not exist. "
                    "Downloading the remote huggingface model: THU-KEG/AdaptThink-7B-delta0.05"
                )
                self.config['model'] = 'THU-KEG/AdaptThink-7B-delta0.05'
            else:
                print(
                    f"Using specified local model: '{self.config['model']}'. "
                )

        if "batch_size" not in self.config:
            print("Warning: No batch_size specified. Using default value of 1.")
            self.config['batch_size'] = 1
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

    def _setup(self):
        try:
            os.environ["VLLM_USE_V1"] = "0"

            self.model = LLM(
                model=self.config['model'],
                enforce_eager=True,
                disable_custom_all_reduce=True,
                trust_remote_code=True,
                gpu_memory_utilization=0.9,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model'],
                cache_dir='./cache',
                trust_remote_code=True
            )
        except Exception as e:
            print(
                f"Warning: Failed to load model from remote. Error: {e}")
            print("Loading THU-KEG/AdaptThink-7B-delta0.05 model.")

            os.environ["VLLM_USE_V1"] = "0"

            self.model = LLM(
                model='THU-KEG/AdaptThink-7B-delta0.05',
                enforce_eager=True,
                disable_custom_all_reduce=True,
                trust_remote_code=True,
                gpu_memory_utilization=0.9,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                'THU-KEG/AdaptThink-7B-delta0.05',
                cache_dir='./cache',
                trust_remote_code=True
            )

        print("Setting up ThinkProbScorer successfully")

    def score_item(self, data_item):
        pass

    def _build_prompt(self, instruction: str) -> str:
        messages = [{"role": "user", "content": instruction}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _score_batch_texts(self, batch_texts: List[str]) -> List[float]:
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            stop=['</think>'],
            include_stop_str_in_output=True,
            logprobs=20,
        )
        outputs = self.model.generate(batch_texts, sampling_params)
        scores = []
        for output in outputs:
            if 151649 not in output.outputs[0].logprobs[0]:
                prob_eot = 0
            else:
                logprob_eot = output.outputs[0].logprobs[0][151649]
                prob_eot = math.exp(logprob_eot.logprob)
            scores.append(round(1 - prob_eot, 3))
        return scores

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        batch_size = self.config['batch_size']
        results: List[Dict] = []

        buf_items, buf_texts = [], []

        with open(dataset, 'r', encoding='utf-8', errors='ignore') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get('name', 'ThinkingProbScorer'))
            for line in f:
                line = line.strip()
                if not line:
                    pbar.update(1)
                    continue
                item = json.loads(line)
                buf_items.append(item)
                buf_texts.append(self._build_prompt(item['instruction']))

                if len(buf_items) == batch_size:
                    batch_scores = self._score_batch_texts(buf_texts)
                    results.extend(
                        {"id": it['id'], "score": sc}
                        for it, sc in zip(buf_items, batch_scores)
                    )
                    buf_items.clear()
                    buf_texts.clear()
                pbar.update(1)

            if buf_items:
                batch_scores = self._score_batch_texts(buf_texts)
                results.extend(
                    {"id": it['id'], "score": sc}
                    for it, sc in zip(buf_items, batch_scores)
                )
                buf_items.clear()
                buf_texts.clear()
            pbar.close()

        return results

    def evaluate_to_file(self, dataset: str, output_file: str, resume: bool = True) -> str:
        num_lines = get_total_lines(dataset)
        batch_size = self.config.get('batch_size', 1)

        done_ids: set = set()
        if resume and os.path.exists(output_file):
            repair_trailing_incomplete_jsonl(output_file)
            done_ids = load_jsonl_id_set(output_file, id_key="id")
            if done_ids:
                print(
                    f"[ThinkingProbScorer] Resume enabled. Found {len(done_ids)} unique completed ids in {output_file}."
                )

        if not resume:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, "w", encoding="utf-8"):
                pass

        buf_items, buf_texts, buf_ids = [], [], []

        with open(dataset, 'r', encoding='utf-8', errors='ignore') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get('name', 'ThinkingProbScorer'))

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
                buf_texts.append(self._build_prompt(item['instruction']))
                buf_ids.append(item.get("id", ""))

                if len(buf_items) == batch_size:
                    batch_scores = self._score_batch_texts(buf_texts)
                    records = [{"id": _id, "score": sc} for _id, sc in zip(buf_ids, batch_scores)]
                    append_jsonl(records, output_file, flush=True)
                    for _id in buf_ids:
                        nid = normalize_id(_id)
                        if nid:
                            done_ids.add(nid)
                    buf_items.clear()
                    buf_texts.clear()
                    buf_ids.clear()
                pbar.update(1)

            if buf_items:
                batch_scores = self._score_batch_texts(buf_texts)
                records = [{"id": _id, "score": sc} for _id, sc in zip(buf_ids, batch_scores)]
                append_jsonl(records, output_file, flush=True)
                for _id in buf_ids:
                    nid = normalize_id(_id)
                    if nid:
                        done_ids.add(nid)
                buf_items.clear()
                buf_texts.clear()
                buf_ids.clear()

            pbar.close()

        return output_file
