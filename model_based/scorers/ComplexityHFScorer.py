import json
import os
import re
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_scorer import BaseScorer
from .utils import get_total_lines
from utils.utils_jsonl import (
    append_jsonl,
    load_jsonl_id_set,
    normalize_id,
    repair_trailing_incomplete_jsonl,
)

COMPLEXITY_PROMPT_TEMPLATE = """\
## role
- You are a rigorous reviewer who is responsible for evaluating the complexity of the 'instruction' of an instruction-output pair.

## goal
- For the given 'instruction', you need to evaluate it according to the evaluation dimension specified in the 'rule': **Complexity**.

## rule
- **Complexity**: whether the 'instruction' requires a certain depth of knowledge and reasoning to understand and output. Your score should strictly follow the rules below:

    - **Score 9-10 (Excellent)**: The instruction requires expert-level knowledge and deep, multi-step reasoning, making it unoutputable without a profound and comprehensive understanding of the subject. It demands the synthesis of multiple complex concepts, potentially from different fields, to generate novel insights or solve new problems. Answering involves sophisticated analysis, abstract or creative thinking under significant constraints, or navigating complex scenarios with ambiguous conditions and interacting variables.

    - **Score 7-8 (Good)**: The instruction is challenging, requiring specialized knowledge and significant reasoning that goes far beyond simple information retrieval. It demands the application of established principles to analyze complex situations, compare different approaches, or evaluate trade-offs. Answering necessitates following multi-step logical procedures or explaining intricate concepts and the relationships between multiple ideas in detail.

    - **Score 5-6 (Acceptable)**: The instruction requires a degree of reasoning that extends beyond basic fact-finding, and the necessary knowledge is not trivial. Answering involves interpreting and organizing information from one or more sources, rather than just quoting them directly. This may include applying a well-known procedure to a straightforward problem or explaining facts that require some context to understand fully.

    - **Score 3-4 (Poor)**: The instruction is simple and straightforward, requiring minimal reasoning or synthesis. The output is typically a single, easily searchable fact, a basic definition, or a direct piece of information that can be retrieved and stated with little to no manipulation or interpretation.

    - **Score 1-2 (Very Poor)**: The instruction is trivial and requires almost no cognitive effort or specialized knowledge to output. It pertains to extremely common knowledge or is a simple closed-ended or subjective instruction whose output is obvious or un-falsifiable.

## output_format
- Do NOT output a JSON object.
- Output EXACTLY two lines (no extra text, no code fences):
    - Line 1: <bos>{{score}}</eos> where {{score}} is an integer from 1 to 10.
    - Line 2: <bor>{{reason}}</eor> where {{reason}} is a brief explanation for your score.

Instruction:
{instruction}

Your output:
<bos>{{score}}</eos>
<bor>{{reason}}</eor>"""


class ComplexityHFScorer(BaseScorer):
    """
    Instruction complexity scorer using HuggingFace transformers (no vLLM).

    Loads a local instruction-tuned LLM via AutoModelForCausalLM, builds a
    complexity-evaluation prompt for each sample's "instruction" field,
    generates judge responses with model.generate(), and parses an integer
    score (1-10) from <bos>…</eos> tags.
    """

    DEFAULT_MODEL = "/mnt/dhwfile/raise/data-leaderboard/model/gaoxin/Qwen3-8B"

    def _validate_config(self):
        if "model" not in self.config:
            print(f"Warning: No model specified. Using default '{self.DEFAULT_MODEL}'.")
            self.config["model"] = self.DEFAULT_MODEL
        elif not os.path.exists(self.config["model"]):
            print(
                f"Warning: Model path '{self.config['model']}' does not exist. "
                f"Falling back to default '{self.DEFAULT_MODEL}'."
            )
            self.config["model"] = self.DEFAULT_MODEL
        else:
            print(f"Using specified model: '{self.config['model']}'.")

        if (
            "batch_size" not in self.config
            or not isinstance(self.config["batch_size"], int)
            or self.config["batch_size"] <= 0
        ):
            self.config["batch_size"] = 4
            print("Warning: No/invalid batch_size. Using default 4.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

        if "max_length" not in self.config:
            self.config["max_length"] = 4096
            print("Warning: No max_length specified. Using default 4096.")
        if "max_new_tokens" not in self.config:
            self.config["max_new_tokens"] = 512
            print("Warning: No max_new_tokens specified. Using default 512.")
        if "temperature" not in self.config:
            self.config["temperature"] = 0.0
            print("Warning: No temperature specified. Using default 0.0 (greedy).")
        if "top_p" not in self.config:
            self.config["top_p"] = 1.0
            print("Warning: No top_p specified. Using default 1.0.")
        if "top_k" not in self.config:
            self.config["top_k"] = None
            print("Warning: No top_k specified. Using default None (not set).")
        if "min_p" not in self.config:
            self.config["min_p"] = None
            print("Warning: No min_p specified. Using default None (not set).")
        if "enable_thinking" not in self.config:
            self.config["enable_thinking"] = False
            print("Warning: No enable_thinking specified. Using default False.")

        self.config.setdefault("min_score", 1)
        self.config.setdefault("max_score", 10)

    def _setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            load_dtype = torch.bfloat16
        elif self.device.type == "cuda":
            load_dtype = torch.float16
        else:
            load_dtype = torch.float32

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model"], trust_remote_code=True, use_fast=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model"],
                trust_remote_code=True,
                torch_dtype=load_dtype,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to set up ComplexityHFScorer with model={self.config['model']}: {e}"
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        self.model.to(self.device)
        self.model.eval()

        self._printed_first_prompt = False
        print(f"Setting up ComplexityHFScorer successfully on {self.device} (dtype={load_dtype})")

    def _build_prompt(self, item: Dict[str, Any]) -> str:
        instruction = item.get("instruction", "")
        if "input" in item and item["input"]:
            instruction = instruction + "\n" + item["input"]
        return COMPLEXITY_PROMPT_TEMPLATE.format(instruction=instruction)

    def _to_chat_prompt(self, user_text: str) -> str:
        enable_thinking = bool(self.config.get("enable_thinking", False))
        messages = [{"role": "user", "content": user_text}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                if not enable_thinking:
                    messages[0]["content"] = user_text + "\n/no_think"
                try:
                    return self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    pass
            except Exception:
                pass
        return user_text

    _SCORE_RE = re.compile(r"<bos>\s*(\d+(?:\.\d+)?)\s*(?:</eos>|</bos>)", re.IGNORECASE)

    def _parse_score(self, text: str) -> Optional[float]:
        if not text:
            return None
        m = self._SCORE_RE.search(text)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    def _clamp_score(self, score: Optional[float]) -> Optional[float]:
        if score is None:
            return None
        mn = self.config.get("min_score")
        mx = self.config.get("max_score")
        if mn is not None:
            score = max(float(mn), score)
        if mx is not None:
            score = min(float(mx), score)
        return float(score)

    def score_item(self, data_item: Dict) -> Dict[str, Any]:
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[Dict[str, Any]]:
        """Return a list of dicts with keys: score, raw_output."""
        if not data_items:
            return []

        prompts: List[str] = []
        for idx, item in enumerate(data_items):
            user_text = self._build_prompt(item)
            final_prompt = self._to_chat_prompt(user_text)
            prompts.append(final_prompt)

            if not self._printed_first_prompt and idx == 0:
                self._printed_first_prompt = True
                try:
                    print("\n" + "=" * 80)
                    print("[ComplexityHFScorer] First-sample debug: final prompt")
                    print("-" * 80)
                    print(final_prompt)
                    print("=" * 80 + "\n")
                except Exception:
                    pass

        encodings = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config["max_length"],
        )
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        input_len = input_ids.shape[1]

        temperature = float(self.config["temperature"])
        do_sample = temperature > 0

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(self.config["max_new_tokens"]),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = float(self.config["top_p"])
            if self.config["top_k"] is not None:
                gen_kwargs["top_k"] = int(self.config["top_k"])
            if self.config["min_p"] is not None:
                gen_kwargs["min_p"] = float(self.config["min_p"])

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        results: List[Dict[str, Any]] = []
        for i in range(len(data_items)):
            new_tokens = output_ids[i][input_len:]
            raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            score = self._clamp_score(self._parse_score(raw_output))
            results.append({"score": score, "raw_output": raw_output})

        return results

    def _flush_batch(self, buf_items, buf_ids):
        batch_results = self.score_batch(buf_items)
        return [
            {"id": _id, "score": r["score"], "raw_output": r["raw_output"]}
            for _id, r in zip(buf_ids, batch_results)
        ]

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []
        batch_size = self.config["batch_size"]
        buf_items: List[Dict] = []
        buf_ids: List[Any] = []

        with open(dataset, "r", encoding="utf-8", errors="ignore") as f:
            pbar = tqdm(total=num_lines, desc=self.config.get("name", "ComplexityHFScorer"))
            for line in f:
                line = line.strip()
                if not line:
                    pbar.update(1)
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    pbar.update(1)
                    continue

                buf_items.append(item)
                buf_ids.append(item.get("id", ""))

                if len(buf_items) == batch_size:
                    results.extend(self._flush_batch(buf_items, buf_ids))
                    buf_items.clear()
                    buf_ids.clear()
                pbar.update(1)

            if buf_items:
                results.extend(self._flush_batch(buf_items, buf_ids))
                buf_items.clear()
                buf_ids.clear()
            pbar.close()

        return results

    def evaluate_to_file(self, dataset: str, output_file: str, resume: bool = True) -> str:
        num_lines = get_total_lines(dataset)
        batch_size = self.config["batch_size"]

        done_ids = set()
        if resume and os.path.exists(output_file):
            repair_trailing_incomplete_jsonl(output_file)
            done_ids = load_jsonl_id_set(output_file, id_key="id")
            if done_ids:
                print(
                    f"[ComplexityHFScorer] Resume enabled. Found {len(done_ids)} "
                    f"unique completed ids in {output_file}."
                )

        if not resume:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8"):
                pass

        buf_items: List[Dict] = []
        buf_ids: List[Any] = []

        with open(dataset, "r", encoding="utf-8", errors="ignore") as f:
            pbar = tqdm(total=num_lines, desc=self.config.get("name", "ComplexityHFScorer"))

            for line in f:
                line = line.strip()
                if not line:
                    pbar.update(1)
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    pbar.update(1)
                    continue

                item_id = normalize_id(item.get("id", ""))
                if item_id and item_id in done_ids:
                    pbar.update(1)
                    continue

                buf_items.append(item)
                buf_ids.append(item.get("id", ""))

                if len(buf_items) == batch_size:
                    records = self._flush_batch(buf_items, buf_ids)
                    append_jsonl(records, output_file, flush=True)
                    for _id in buf_ids:
                        nid = normalize_id(_id)
                        if nid:
                            done_ids.add(nid)
                    buf_items.clear()
                    buf_ids.clear()
                pbar.update(1)

            if buf_items:
                records = self._flush_batch(buf_items, buf_ids)
                append_jsonl(records, output_file, flush=True)
                for _id in buf_ids:
                    nid = normalize_id(_id)
                    if nid:
                        done_ids.add(nid)
                buf_items.clear()
                buf_ids.clear()

            pbar.close()

        return output_file
