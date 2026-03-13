import torch
from .base_scorer import BaseScorer
import json
import math
import os
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from .utils import get_total_lines
import warnings
from utils.utils_jsonl import append_jsonl, repair_trailing_incomplete_jsonl, load_jsonl_id_set, normalize_id


class NormLossScorer(BaseScorer):
    def _validate_config(self):
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model.")
            self.config['model'] = 'meta-llama/Llama-3.1-8B'
        else:
            print(f"Using specified local model: '{self.config['model']}'. ")

        if "max_length" in self.config and isinstance(self.config["max_length"], int) and self.config["max_length"] > 0:
            print(f"Using specified max_length: {self.config['max_length']}.")
        elif "max_length" in self.config and isinstance(self.config["max_length"], int) and self.config["max_length"] <= 0:
            print(
                "Warning: the specific max_length should > 0. use default value of 2048.")
            self.config['max_length'] = 2048
        else:
            print("Warning: No specific max_length, use default value of 2048.")
            self.config['max_length'] = 2048

        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 8
            print("Warning: No/invalid batch_size, use default value of 8.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

    def _setup(self):
        # These outputs are NOT needed for normalized-loss scoring and can significantly increase memory usage.
        output_hidden_states = bool(self.config.get("output_hidden_states", False))
        output_attentions = bool(self.config.get("output_attentions", False))

        # KV cache is useful for generation, but not needed for full-sequence NLL computation and can increase memory.
        use_cache = bool(self.config.get("use_cache", False))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model"],
                device_map="auto",
                torch_dtype="auto",
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"])
        except Exception as e:
            print(
                f"Load specified model failed ({e}), fall back to meta-llama/Llama-3.1-8B")
            self.model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.1-8B",
                device_map="auto",
                torch_dtype="auto",
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.1-8B")

        self.model.eval()

        # Ensure pad_token exists for batch padding.
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            try:
                if getattr(self.model.config, "pad_token_id", None) is None:
                    self.model.config.pad_token_id = self.tokenizer.eos_token_id
            except Exception:
                pass

        # Ensure model config matches our scoring usage.
        try:
            self.model.config.use_cache = use_cache
        except Exception:
            pass

        # Clamp max_length to the model's supported context length if available.
        try:
            model_max_pos = getattr(self.model.config, "max_position_embeddings", None)
            if isinstance(model_max_pos, int) and model_max_pos > 0:
                if int(self.config.get("max_length", 0) or 0) > model_max_pos:
                    print(
                        f"Warning: config.max_length ({self.config['max_length']}) > model.max_position_embeddings "
                        f"({model_max_pos}). Clamping max_length to {model_max_pos} to avoid OOM/invalid positions."
                    )
                    self.config["max_length"] = model_max_pos
        except Exception:
            pass

        print(f"Setting up NormLossScorer successfully on {self.device}")

    def score_item(self, data_item):
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[float]:
        texts = []
        for item in data_items:
            instruction=item["instruction"]
            input_text=item.get("input_text", "")
            response = item["output"]
            if input_text:
                input_text = instruction + '\n' + input_text + '\n' + response
            else:
                input_text = instruction + '\n' + response
            texts.append(input_text)

        # Batch tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.config["max_length"]
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Check for truncated sequences and issue warning
        max_seq_length = attention_mask.sum(dim=1).max().item()
        if max_seq_length >= self.config["max_length"]:
            truncated_count = (attention_mask.sum(dim=1) >= self.config["max_length"]).sum().item()
            warnings.warn(
                f"Warning: {truncated_count} out of {len(data_items)} sequences were truncated "
                f"to max_length={self.config['max_length']}. Consider increasing max_length for complete processing.",
                UserWarning
            )

        # Create labels, set padding positions to -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        scores = []
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            
            # Get loss for each token (using natural logarithm)
            # Use ignore_index=-100 to ignore padding tokens
            loss_per_token = torch.nn.functional.cross_entropy(
                outputs.logits[:, :-1, :].contiguous().view(-1, outputs.logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                reduction='none',
                ignore_index=-100
            )

            # Reshape to [batch_size, seq_len-1]
            loss_per_token = loss_per_token.view(input_ids.size(0), -1)
            # Adjust attention_mask to match loss_per_token shape (remove first token)
            valid_mask = attention_mask[:, 1:].float()

            # Calculate normalized loss for each sample
            for i in range(input_ids.size(0)):
                # Use valid_mask to calculate loss for valid tokens
                # Note: padding positions in loss_per_token are already handled by ignore_index, returning 0
                masked_loss = loss_per_token[i] * valid_mask[i]
                total_loss = masked_loss.sum().item()
                total_tokens = valid_mask[i].sum().item()

                # Normalize and convert from natural logarithm to log2
                if total_tokens > 0:
                    normalized_loss = (total_loss / total_tokens) / math.log(2)
                else:
                    normalized_loss = 0.0
                
                scores.append(normalized_loss)

        return scores

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config.get("batch_size")
        buffer_items = []
        buffer_ids = []

        with open(dataset, 'r') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get(
                'name', 'NormLossScorer'))
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
                    pbar.update(len(buffer_items))
                    buffer_items.clear()
                    buffer_ids.clear()

            if buffer_items:
                batch_scores = self.score_batch(buffer_items)
                results.extend([
                    {"id": id_, "score": sc}
                    for id_, sc in zip(buffer_ids, batch_scores)
                ])
                pbar.update(len(buffer_items))
                buffer_items.clear()
                buffer_ids.clear()
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
                print(f"[NormLossScorer] Resume enabled. Found {len(done_ids)} unique completed ids in {output_file}.")

        if not resume:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8"):
                pass

        buf_items, buf_ids = [], []

        with open(dataset, "r", encoding="utf-8", errors="ignore") as f:
            pbar = tqdm(total=num_lines, desc=self.config.get("name", "NormLossScorer"))

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

