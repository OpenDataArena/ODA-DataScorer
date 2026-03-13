# import re
import torch
import torch.nn.functional as F
from .base_scorer import BaseScorer
import json
from typing import Dict, List
from transformers import AutoTokenizer
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from .utils import get_total_lines
import json
from tqdm import tqdm
from utils.utils_jsonl import append_jsonl, repair_trailing_incomplete_jsonl, load_jsonl_id_set, normalize_id


class IFDScorer(BaseScorer):
    def _validate_config(self):
        if "model" not in self.config:
            print(
                "Warning: No loacl model specified in config. Downloading the remote huggingface model.")
            self.config['model'] = 'openai-community/gpt2'
        else:
            if not os.path.exists(self.config["model"]):
                print(
                    f"Warning: Specified local model path '{self.config['model']}' does not exist. "
                    "Downloading the remote huggingface model: openai-community/gpt2"
                )
                self.config['model'] = 'openai-community/gpt2'
            else:
                print(
                    f"Using specified local model: '{self.config['model']}'. "
                )

        if "max_length" in self.config and isinstance(self.config["max_length"], int) and self.config["max_length"] > 0:
            print(
                f"Using specified max_length: {self.config['max_length']}.")
        else:
            print(
                "Warning: No specific max_length, use default value of 2048.")
            self.config['max_length'] = 2048

        if "batch_size" in self.config and isinstance(self.config["batch_size"], int) and self.config["batch_size"] > 0:
            print(
                f"Using specified batch_size: {self.config['batch_size']}.")
        else:
            print(
                "Warning: No specific batch_size, use default value of 1.")
            self.config['batch_size'] = 1

        if "template" in self.config and isinstance(self.config["template"], str):
            print(
                f"Using specified template: {self.config['template']}.")
        else:
            print(
                "Warning: No specific template, use default value of qwen2.")
            self.config['template_no_input'] = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
            self.config['template'] = "<|im_start|>user\n{instruction}\n{input}<|im_end|>\n<|im_start|>assistant\n"

    def _setup(self):
        # These outputs are NOT needed for perplexity/IFD scoring and can significantly increase memory usage.
        output_hidden_states = bool(self.config.get("output_hidden_states", False))
        output_attentions = bool(self.config.get("output_attentions", False))

        # KV cache is useful for generation, but not needed for full-sequence NLL computation and can increase memory.
        use_cache = bool(self.config.get("use_cache", False))

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['model'],
                device_map="auto",
                cache_dir='../cache',
                torch_dtype="auto",
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model'], cache_dir='./cache')
        except Exception:
            print(
                "Warning: Failed to load model from remote. Loading openai-community/gpt2 model.")
            self.model = AutoModelForCausalLM.from_pretrained(
                'openai-community/gpt2',
                device_map="auto",
                cache_dir='../cache',
                torch_dtype="auto",
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                'openai-community/gpt2', cache_dir='./cache')
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

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            print("Warning: No GPU available. Using CPU.")
            self.device = "cpu"
        print("Setting up IFDScorer successfully")

    def _batch_ppl(self, texts: List[str], max_length: int):
        """
        True batched PPL computation (per-sample), using one forward for the whole batch.
        Returns: (perplexities, losses)
        """
        if len(texts) == 0:
            return [], []
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

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

        # Per-token loss (flatten then reshape)
        loss_flat = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )
        loss_tok = loss_flat.view(shift_labels.size(0), -1)

        valid = (shift_labels != -100)
        denom = valid.sum(dim=1).clamp_min(1)
        loss_sum = (loss_tok * valid.float()).sum(dim=1)
        loss_per_sample = loss_sum / denom
        ppl_per_sample = torch.exp(loss_per_sample)

        return ppl_per_sample.detach().cpu().tolist(), loss_per_sample.detach().cpu().tolist()

    def _batch_ppl_from_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Batched PPL from already-tokenized/padded ids.
        input_ids: [B, T], attention_mask: [B, T]
        Returns: (perplexities, losses)
        """
        if input_ids.numel() == 0:
            return [], []

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits

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

        return ppl_per_sample.detach().cpu().tolist(), loss_per_sample.detach().cpu().tolist()

    def _batch_conditional_ppl(self, whole_texts: List[str], prefix_lens: List[int]):
        """
        True batched conditional PPL: compute NLL only on tokens after prefix_lens[i].
        Returns: (perplexities, losses)
        """
        if len(whole_texts) == 0:
            return [], []

        enc = self.tokenizer(
            whole_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config["max_length"],
        ).to(self.device)

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()

        # Mask out prefix tokens per sample
        for i, pl in enumerate(prefix_lens):
            pl = int(pl or 0)
            if pl > 0:
                pl = min(pl, labels.size(1))
                labels[i, :pl] = -100
        # Mask padding
        labels[attention_mask == 0] = -100

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits

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
        # If denom==0, no valid tokens (e.g., prefix covers all or truncation). Mark as invalid.
        loss_sum = (loss_tok * valid.float()).sum(dim=1)
        loss_per_sample = torch.zeros_like(loss_sum)
        ok = denom > 0
        loss_per_sample[ok] = loss_sum[ok] / denom[ok].float()
        ppl_per_sample = torch.zeros_like(loss_per_sample)
        ppl_per_sample[ok] = torch.exp(loss_per_sample[ok])

        return ppl_per_sample.detach().cpu().tolist(), loss_per_sample.detach().cpu().tolist()

    def _batch_conditional_ppl_from_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, prefix_lens: List[int]):
        """
        Batched conditional PPL from already-tokenized/padded ids.
        Only tokens AFTER prefix_lens[i] contribute to loss for sample i.
        Returns: (perplexities, losses)
        """
        if input_ids.numel() == 0:
            return [], []

        labels = input_ids.clone()
        # Mask out prefix tokens per sample (prefix_len counted on the SAME tokenization as input_ids)
        for i, pl in enumerate(prefix_lens):
            pl = int(pl or 0)
            if pl > 0:
                pl = min(pl, labels.size(1))
                labels[i, :pl] = -100
        # Mask padding
        labels[attention_mask == 0] = -100

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits

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

        return ppl_per_sample.detach().cpu().tolist(), loss_per_sample.detach().cpu().tolist()

    def _pad_id_seqs(self, seqs: List[List[int]]) -> (torch.Tensor, torch.Tensor):
        """
        Pad a list of token-id sequences into (input_ids, attention_mask).
        """
        if len(seqs) == 0:
            empty = torch.empty((0, 0), dtype=torch.long, device=self.device)
            return empty, empty
        max_len = max(len(s) for s in seqs)
        pad_id = int(getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", 0) or 0)
        input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            if not s:
                continue
            l = len(s)
            input_ids[i, :l] = torch.tensor(s, dtype=torch.long, device=self.device)
            attention_mask[i, :l] = 1
        return input_ids, attention_mask

    def get_perplexity_and_embedding_whole_text(self, text, max_length):
        try:
            # Tokenize without truncation first to check length
            full_tokens = self.tokenizer.encode(text, return_tensors="pt")
            if full_tokens.shape[1] > max_length:
                print(f"Warning: Text length ({full_tokens.shape[1]} tokens) exceeds max_length ({max_length}), truncating.")
            
            input_ids = self.tokenizer.encode(
                text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )
            perplexity = torch.exp(loss)

            return perplexity.to('cpu').item(), loss.to('cpu').item()
        except Exception as e:
            print(f"Error in get_perplexity_and_embedding_whole_text: {e}")
            return 0, 0

    def get_perplexity_and_embedding_part_text(self, whole_text: str, prefix_len: int):
        try:
            input_ids = self.tokenizer.encode(
                whole_text, return_tensors="pt", truncation=True, max_length=self.config['max_length']
            ).to(self.device)
            labels = input_ids.clone()
            prefix_len = int(prefix_len or 0)
            if prefix_len > 0:
                prefix_len = min(prefix_len, labels.shape[1])
                labels[0, :prefix_len] = -100

            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
                ignore_index=-100,
            )
            perplexity = torch.exp(loss)
            return perplexity.to('cpu').item(), loss.to('cpu').item()
        except Exception as e:
            print(f"Error in get_perplexity_and_embedding_part_text: {e}")
            return 0, 0

    def get_batch_perplexity_whole_text(self, texts, max_length):
        """Compute perplexity for a batch of texts"""
        try:
            return self._batch_ppl(texts, max_length=max_length)
        except Exception as e:
            print(f"Error in get_batch_perplexity_whole_text: {e}")
            return [0] * len(texts), [0] * len(texts)

    def get_batch_perplexity_part_text(self, texts, target_spans):
        """Compute conditional perplexity for a batch of texts (legacy API, kept for compatibility)."""
        try:
            # Fallback: compute prefix_lens by locating target_spans in raw text.
            prefix_lens = []
            for text, target_span in zip(texts, target_spans):
                start_index = str(text).rfind(str(target_span))
                if start_index < 0:
                    prefix_lens.append(0)
                else:
                    prefix_lens.append(len(self.tokenizer.encode(text[:start_index])))
            return self._batch_conditional_ppl(texts, prefix_lens=prefix_lens)
        except Exception as e:
            print(f"Error in get_batch_perplexity_part_text: {e}")
            return [0] * len(texts), [0] * len(texts)

    def score_item(self, data_item):
        # Delegate to the batch implementation (B=1) to guarantee consistent token-boundary handling.
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items):
        """
        Score multiple data items in true batch (one forward for the whole provided list).
        Note: peak memory depends on (max_length, batch_size, data length distribution). Users can
        control it by setting batch_size/max_length in config.
        """
        scores = [-1] * len(data_items)

        batch_outputs = []
        batch_whole_texts = []
        batch_prefix_lens = []
        valid_indices = []
        
        for idx, data_item in enumerate(data_items):
            instruction = data_item["instruction"]
            _input = data_item.get("input", "")
            output = data_item["output"]
            
            if output is None or (isinstance(output, str) and len(output) == 0):
                print(f"data_item's output is empty: {data_item}, return -1")
                continue
            
            if _input is None or (isinstance(_input, str) and len(_input) == 0):
                prompt = self.config['template_no_input'].format(instruction=instruction)
            else:
                prompt = self.config['template'].format(instruction=instruction, input=_input)
            
            # IMPORTANT: compute prefix_len on the SAME tokenization as the concatenated sequence.
            # We do this by separately tokenizing prompt/output (no special tokens) and concatenating ids.
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
            output_ids = self.tokenizer(output, add_special_tokens=False).input_ids

            whole_ids = prompt_ids + output_ids
            max_len = int(self.config["max_length"])
            if max_len > 0 and len(whole_ids) > max_len:
                whole_ids = whole_ids[:max_len]

            prefix_len = min(len(prompt_ids), len(whole_ids))
            out_kept = max(0, len(whole_ids) - prefix_len)

            # Keep unconditional output length aligned with what actually survives inside (prompt+output) truncation.
            output_ids_kept = output_ids[:out_kept]

            # Add BOS token to output for unconditional PPL calculation.
            # This ensures the first output token's loss is computed, matching conditional PPL.
            bos_id = self.tokenizer.bos_token_id
            if bos_id is None:
                bos_id = self.tokenizer.eos_token_id or 0
            output_ids_with_bos = [bos_id] + output_ids_kept

            batch_outputs.append(output_ids_with_bos)
            batch_whole_texts.append(whole_ids)
            batch_prefix_lens.append(prefix_len)
            valid_indices.append(idx)
        
        if len(batch_outputs) == 0:
            return scores
        
        try:
            # Pad to tensors and compute ppl in true batch, using consistent token boundaries.
            whole_input_ids, whole_attn = self._pad_id_seqs(batch_whole_texts)
            out_input_ids, out_attn = self._pad_id_seqs(batch_outputs)

            ppl_out_alone_list, _ = self._batch_ppl_from_ids(out_input_ids, out_attn)
            ppl_out_condition_list, _ = self._batch_conditional_ppl_from_ids(
                whole_input_ids, whole_attn, prefix_lens=batch_prefix_lens
            )

            for i, idx in enumerate(valid_indices):
                if ppl_out_alone_list[i] > 0 and ppl_out_condition_list[i] > 0:
                    scores[idx] = ppl_out_condition_list[i] / ppl_out_alone_list[i]
                else:
                    scores[idx] = -1
            return scores
        except Exception as e:
            print(f"Error in score_batch: {e}")
            return scores

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        results = []
        batch_size = self.config.get('batch_size', 1)
        
        if batch_size == 1:
            # Single sample processing mode (maintain backward compatibility)
            with open(dataset, 'r') as f:
                for line in tqdm(f, total=num_lines, desc=self.config['name']):
                    item = json.loads(line.strip())
                    res = {
                        "id": item.get("id", ""),
                        "score": self.score_item(item)
                    }
                    results.append(res)
        else:
            # Batch processing mode
            with open(dataset, 'r') as f:
                batch_items = []
                batch_ids = []
                
                for line in tqdm(f, total=num_lines, desc=self.config['name']):
                    item = json.loads(line.strip())
                    batch_items.append(item)
                    batch_ids.append(item.get("id", ""))
                    
                    if len(batch_items) >= batch_size:
                        # Process current batch
                        scores = self.score_batch(batch_items)
                        for item_id, score in zip(batch_ids, scores):
                            results.append({
                                "id": item_id,
                                "score": score
                            })
                        batch_items = []
                        batch_ids = []
                
                # Process the last incomplete batch
                if batch_items:
                    scores = self.score_batch(batch_items)
                    for item_id, score in zip(batch_ids, scores):
                        results.append({
                            "id": item_id,
                            "score": score
                        })
        
        return results

    def evaluate_to_file(self, dataset: str, output_file: str, resume: bool = True) -> str:
        """
        Stream pointwise results to JSONL (append), enabling resume from existing output_file.
        Assumption: one output JSON line per input JSON line, and order is consistent.
        """
        num_lines = get_total_lines(dataset)
        batch_size = int(self.config.get("batch_size", 1) or 1)

        done_ids = set()
        if resume and os.path.exists(output_file):
            repair_trailing_incomplete_jsonl(output_file)
            done_ids = load_jsonl_id_set(output_file, id_key="id")
            if done_ids:
                print(f"[IFDScorer] Resume enabled. Found {len(done_ids)} unique completed ids in {output_file}.")

        if not resume:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8"):
                pass

        with open(dataset, "r", encoding="utf-8", errors="ignore") as f:
            pbar = tqdm(total=num_lines, desc=self.config.get("name", "IFDScorer"))

            if batch_size <= 1:
                buffer_records = []
                flush_every = int(self.config.get("flush_every", 1) or 1)

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
                    record = {"id": item.get("id", ""), "score": self.score_item(item)}
                    buffer_records.append(record)

                    if len(buffer_records) >= flush_every:
                        append_jsonl(buffer_records, output_file, flush=True)
                        for rec in buffer_records:
                            rid = normalize_id(rec.get("id"))
                            if rid:
                                done_ids.add(rid)
                        buffer_records.clear()

                    pbar.update(1)

                if buffer_records:
                    append_jsonl(buffer_records, output_file, flush=True)
                    for rec in buffer_records:
                        rid = normalize_id(rec.get("id"))
                        if rid:
                            done_ids.add(rid)
                    buffer_records.clear()
            else:
                batch_items = []
                batch_ids = []

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
                    batch_items.append(item)
                    batch_ids.append(item.get("id", ""))

                    if len(batch_items) >= batch_size:
                        scores = self.score_batch(batch_items)
                        records = [{"id": _id, "score": sc} for _id, sc in zip(batch_ids, scores)]
                        append_jsonl(records, output_file, flush=True)
                        for _id in batch_ids:
                            rid = normalize_id(_id)
                            if rid:
                                done_ids.add(rid)
                        batch_items = []
                        batch_ids = []

                    pbar.update(1)

                if batch_items:
                    scores = self.score_batch(batch_items)
                    records = [{"id": _id, "score": sc} for _id, sc in zip(batch_ids, scores)]
                    append_jsonl(records, output_file, flush=True)
                    for _id in batch_ids:
                        rid = normalize_id(_id)
                        if rid:
                            done_ids.add(rid)
                    batch_items = []
                    batch_ids = []

            pbar.close()

        return output_file
