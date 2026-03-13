import json
import os
import math
from typing import Dict, List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from .base_scorer import BaseScorer
from .utils import get_total_lines
from utils.utils_jsonl import (
    append_jsonl,
    load_jsonl_id_set,
    normalize_id,
    repair_trailing_incomplete_jsonl,
)
import numpy as np


class AskLlmScorer(BaseScorer):
    """
    Score data quality using a specified LLM and prompt.
    Computes the average log probability of yes_token, i.e., mean(log P(token_i | prompt, data)).
    For single-token yes_token, returns its log probability; for multi-token yes_token, returns the average of all token log probabilities.
    """

    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate model path
        if "model" not in self.config:
            print("Warning: No model specified. Use default 'Qwen/Qwen2.5-7B'.")
            self.config["model"] = "Qwen/Qwen2.5-7B"


        # Validate prompt
        if "prompt" not in self.config:
            print("Warning: No prompt specified. Use default prompt.")
            self.config["prompt"] = "Is the following data high quality? Please answer yes or no.\n\n"
        else:
            print(f"Using specified prompt: {self.config['prompt'][:50]}...")

        # Validate yes_token
        if "yes_token" not in self.config:
            self.config["yes_token"] = "yes"
            print("Warning: No yes_token specified. Use default 'yes'.")
        else:
            print(f"Using specified yes_token: {self.config['yes_token']}")

        # Validate batch_size
        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 8
            print("Warning: No/invalid batch_size, use default value of 8.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

        # Validate max_length
        if "max_length" not in self.config:
            self.config["max_length"] = 2048
            print("Warning: No max_length specified, use default value of 2048.")
        else:
            print(f"Using specified max_length: {self.config['max_length']}.")

    def _setup(self):
        """Load model and tokenizer"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Determine optimal dtype: use mixed precision strategy
        # Model loading uses bfloat16/float16 to save memory, critical computations convert to float32 for precision
        if self.device == "cuda":
            # Prefer bfloat16 (better numerical stability, larger dynamic range)
            if torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
                print("Loading model with bfloat16 precision (mixed precision: model bfloat16, computation float32)")
            else:
                # Fall back to float16 if bfloat16 is not supported
                torch_dtype = torch.float16
                print("Loading model with float16 precision (mixed precision: model float16, computation float32)")
        else:
            # Use float32 for CPU mode
            torch_dtype = torch.float32
            print("Loading model with float32 precision (CPU mode)")
        
        # Allow config file to override dtype
        if "model_dtype" in self.config:
            dtype_map = {
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
                "float16": torch.float16
            }
            if self.config["model_dtype"] in dtype_map:
                torch_dtype = dtype_map[self.config["model_dtype"]]
                print(f"Config override: using {self.config['model_dtype']} precision")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['model'],
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model'])
            
            # Set pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{self.config['model']}': {e}")


        if self.device == "cuda":
            self.model.to(self.device)
        
        self.model.eval()
        
        print(f"Setting up AskLlmScorer successfully")
        print(f"Device: {self.device}")
        print(f"Model dtype: {torch_dtype}")

    def score_item(self, data_item: Dict) -> float:
        """Score a single sample"""
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[float]:
        """Batch parallel computation of yes_token's average log probability"""
        if len(data_items) == 0:
            return []
        
        # Prepare batch data
        prompts = []
        full_texts = []
        original_lengths = []  # Track original text lengths for truncation detection
        
        for item in data_items:
            instruction = item["instruction"]
            input_text = item.get("input", "")
            output = item["output"]
            
            # Construct data text
            if input_text:
                data_text = instruction + "\n" + input_text + "\n" + output + "\n"
            else:
                data_text = instruction + "\n" + output + "\n"
            
            # Construct complete prompt
            prompt = self.config["prompt"] + data_text + "\n\n"
            prompts.append(prompt)
            
            # Complete text = prompt + yes_token
            full_text = prompt + self.config["yes_token"]
            full_texts.append(full_text)
            
            # Store original length (in tokens) for truncation detection
            original_token_count = len(self.tokenizer.encode(full_text, add_special_tokens=False))
            original_lengths.append(original_token_count)
        
        # Batch tokenize prompt part (to get prompt length for each sample)
        prompt_tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True
        )
        prompt_lengths = (prompt_tokens.attention_mask.sum(dim=1)).tolist()
        
        # Batch tokenize complete text (prompt + yes_token)
        full_tokens = self.tokenizer(
            full_texts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.config["max_length"]
        )
        
        input_ids = full_tokens.input_ids.to(self.device)
        attention_mask = full_tokens.attention_mask.to(self.device)
        
        batch_size = input_ids.shape[0]
        
        # Check for truncation and issue warnings
        truncated_count = 0
        for idx, original_len in enumerate(original_lengths):
            if original_len > self.config["max_length"]:
                truncated_count += 1
                sample_id = data_items[idx].get("id", f"index_{idx}")
                print(f"WARNING: Sample (id: {sample_id}) has been truncated. "
                      f"Original length: {original_len} tokens, Max length: {self.config['max_length']} tokens. "
                      f"This may affect scoring accuracy.")
        
        if truncated_count > 0:
            print(f"WARNING: {truncated_count} out of {batch_size} samples in this batch were truncated due to max_length limit.")
        
        # Batch forward propagation
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)
        
        # Compute average log probability of yes_token for each sample
        batch_scores = []
        
        for batch_idx in range(batch_size):
            prompt_length = prompt_lengths[batch_idx]
            
            # Find the actual length of the current sample (non-padding part)
            sample_length = attention_mask[batch_idx].sum().item()
            
            # If prompt is too long or no yes_token part, set to a very small negative number
            if prompt_length >= self.config["max_length"] or prompt_length >= sample_length:
                batch_scores.append(-100.0)
                continue
            
            # Calculate log probability for each token in yes_token part
            log_probs = []
            
            # Iterate through each token in yes_token part (starting from prompt_length)
            for i in range(prompt_length, sample_length):
                # Check if it's padding
                if attention_mask[batch_idx, i].item() == 0:
                    break
                
                true_token_id = input_ids[batch_idx, i].item()
                
                # Skip padding token
                if true_token_id == self.tokenizer.pad_token_id:
                    continue
                
                # Get logits for predicting this token (output at position i-1 predicts token at position i)
                current_logits = logits[batch_idx, i - 1, :]
                
                # [CRITICAL] Convert to float32 for improved numerical precision and stability
                # This is especially important when the model uses float16/bfloat16, avoiding numerical underflow
                current_logits = current_logits.float()
                
                # Calculate log probability distribution (using float32 precision)
                log_probs_all = F.log_softmax(current_logits, dim=-1)
                
                # Get log probability of the target token
                token_log_prob = log_probs_all[true_token_id].item()
                log_probs.append(token_log_prob)
            
            # Calculate average log probability
            if log_probs:
                avg_log_prob = np.mean(log_probs)
            else:
                avg_log_prob = -100.0  # If no tokens, set to a very small negative number
            
            batch_scores.append(avg_log_prob)
        
        return batch_scores

    def evaluate(self, dataset: str) -> List[Dict]:
        """Evaluate the entire dataset"""
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config.get("batch_size")
        buffer_items = []
        buffer_ids = []

        with open(dataset, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get(
                'name', 'AskLlmScorer'))
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
                    f"[AskLlmScorer] Resume enabled. Found {len(done_ids)} unique completed ids in {output_file}."
                )

        if not resume:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, "w", encoding="utf-8"):
                pass

        buf_items, buf_ids = [], []

        with open(dataset, "r", encoding="utf-8", errors="ignore") as f:
            pbar = tqdm(total=num_lines, desc=self.config.get("name", "AskLlmScorer"))

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
