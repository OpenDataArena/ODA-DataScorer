import torch
import torch.nn.functional as F
from .base_scorer import BaseScorer
import json
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
from .utils import get_total_lines
from utils.utils_jsonl import append_jsonl, repair_trailing_incomplete_jsonl, load_jsonl_id_set, normalize_id


class PPLScorer(BaseScorer):
    def _validate_config(self):
        # Validate model path
        if "model" not in self.config:
            print("Warning: No model specified, use default model 'Qwen/Qwen3-8B'.")
            self.config['model'] = 'Qwen/Qwen3-8B'
        else:
            print(f"Using specified model: '{self.config['model']}'.")

        # Validate max_length
        if "max_length" in self.config and isinstance(self.config["max_length"], int) and 0 < self.config["max_length"]:
            print(f"Using specified max_length: {self.config['max_length']}.")
        elif "max_length" in self.config and isinstance(self.config["max_length"], int) and self.config["max_length"] <= 0:
            print("Warning: the specific max_length should > 0. use default value of 2048.")
            self.config['max_length'] = 2048
        else:
            print("Warning: No specific max_length, use default value of 2048.")
            self.config['max_length'] = 2048

        # Validate batch_size
        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 8
            print("Warning: No/invalid batch_size, use default value of 8.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

    def _setup(self):
        # These outputs are NOT needed for perplexity scoring and can significantly increase memory usage.
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
            raise RuntimeError(f"Failed to load model: {e}")

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

        print(f"Setting up PPLScorer successfully on {self.device}")

    def score_item(self, data_item: Dict) -> float:
        """Calculate PPL score for a single sample (lower is better)"""
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[float]:
        """
        Calculate PPL scores in batch.
        Critical: Correctly handle padding and attention_mask, ensuring padding tokens in labels are ignored.
        """
        # Extract text content from data items, concatenating instruction, input, and output
        texts = []
        for item in data_items:
            parts = []
            parts.append(item["instruction"])
            parts.append(item.get("input", ""))
            parts.append(item["output"])
            
            # Join with newlines, filtering out empty strings
            text = "\n".join([p for p in parts if p])
            texts.append(text)

        # Batch tokenize with padding and truncation enabled
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,        # Critical: Enable padding
            truncation=True,     # Critical: Enable truncation
            max_length=self.config["max_length"]
        ).to(self.device)

        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask

        # Check for truncation and issue warning if any text was truncated
        max_length = self.config["max_length"]
        for i, text in enumerate(texts):
            # Check if the sequence length reaches max_length (indicating potential truncation)
            seq_length = attention_mask[i].sum().item()
            if seq_length == max_length:
                print(f"Warning: Certain data may have been truncated to max_length={max_length}.")

        # Create labels and ignore padding tokens
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits

        # Shift for causal LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab_size = shift_logits.size(-1)

        loss_flat = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )
        loss_tok = loss_flat.view(shift_labels.size(0), -1)

        valid = (shift_labels != -100)
        denom = valid.sum(dim=1)
        loss_sum = (loss_tok * valid.float()).sum(dim=1)
        loss_per_sample = torch.zeros_like(loss_sum)
        ok = denom > 0
        loss_per_sample[ok] = loss_sum[ok] / denom[ok].float()
        ppl_per_sample = torch.zeros_like(loss_per_sample)
        ppl_per_sample[ok] = torch.exp(loss_per_sample[ok])

        return ppl_per_sample.detach().cpu().tolist()

    def evaluate(self, dataset: str) -> List[Dict]:
        """Evaluate the entire dataset and return PPL scores for each sample"""
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config.get("batch_size")
        buffer_items = []
        buffer_ids = []

        with open(dataset, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get('name', 'PPLScorer'))
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
                pbar.update(batch_size)

            # Process remaining samples
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
        Resume logic is id-based: skip any sample whose id already exists in output_file.
        """
        num_lines = get_total_lines(dataset)
        batch_size = self.config.get("batch_size")

        done_ids = set()
        if resume and os.path.exists(output_file):
            repair_trailing_incomplete_jsonl(output_file)
            done_ids = load_jsonl_id_set(output_file, id_key="id")
            if done_ids:
                print(f"[PPLScorer] Resume enabled. Found {len(done_ids)} unique completed ids in {output_file}.")

        if not resume:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8"):
                pass

        buf_items, buf_ids = [], []

        with open(dataset, "r", encoding="utf-8", errors="ignore") as f:
            pbar = tqdm(total=num_lines, desc=self.config.get("name", "PPLScorer"))

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