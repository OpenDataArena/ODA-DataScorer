import json
import os
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .base_scorer import BaseScorer
from .utils import get_total_lines

try:
    # Most common runtime: run from data_scorer/model_based, so "utils" is the local package.
    from utils.utils_jsonl import append_jsonl, repair_trailing_incomplete_jsonl, load_jsonl_id_set, normalize_id
except Exception:
    # Fallback: when imported as a package module.
    from ..utils.utils_jsonl import append_jsonl, repair_trailing_incomplete_jsonl, load_jsonl_id_set, normalize_id


class EmbedSVDEntropyScorer(BaseScorer):
    """
    Embed the text and take the raw feature matrix of last_hidden_state (L x H);
    Perform SVD on this matrix to obtain the singular values sigma;
    Normalize to obtain p_j = sigma_j / sum(sigma);
    Compute the information entropy V_inf = - sum_j p_j * log(p_j + eps).
    """

    def _validate_config(self):
        if "model" not in self.config:
            print("Warning: No embedding model specified, use default: Qwen/Qwen3-Embedding-8B")
            self.config["model"] = "Qwen/Qwen3-Embedding-8B"

        # Many HF models (esp. Qwen series) require trust_remote_code=True for correct loading.
        if "trust_remote_code" not in self.config or not isinstance(self.config["trust_remote_code"], bool):
            self.config["trust_remote_code"] = True

        # max_length: Embedding models usually support longer context, here a relatively conservative default value is given
        if "max_length" not in self.config or not isinstance(self.config["max_length"], int) or self.config["max_length"] <= 0:
            print("Warning: No/invalid max_length, use default value of 8192.")
            self.config["max_length"] = 8192
        else:
            print(f"Using specified max_length: {self.config['max_length']}.")

        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            # SVD is relatively heavy, so the default batch should not be too large
            self.config["batch_size"] = 8
            print("Warning: No/invalid batch_size, use default value of 8.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

        # SVD on per-token hidden states is very expensive for long sequences.
        # To prevent OOM / extreme slowness, we cap the effective token rows used for SVD.
        if "svd_max_tokens" not in self.config or not isinstance(self.config["svd_max_tokens"], int) or self.config["svd_max_tokens"] <= 0:
            self.config["svd_max_tokens"] = 512
            print("Warning: No/invalid svd_max_tokens, use default value of 512.")
        else:
            print(f"Using specified svd_max_tokens: {self.config['svd_max_tokens']}.")

        # Token selection strategy when sequence length > svd_max_tokens:
        # - tail: keep last svd_max_tokens tokens (recommended for causal embedding models)
        # - uniform: uniformly sample svd_max_tokens positions across the sequence
        if "svd_token_strategy" not in self.config or self.config["svd_token_strategy"] not in ["tail", "uniform"]:
            self.config["svd_token_strategy"] = "tail"
            print("Warning: No/invalid svd_token_strategy, use default value of 'tail'.")
        else:
            print(f"Using specified svd_token_strategy: {self.config['svd_token_strategy']}.")

        if "fields" not in self.config or not isinstance(self.config["fields"], list) or not self.config["fields"]:
            # Align with the default behavior of embed.py: instruction + input + output
            self.config["fields"] = ["instruction", "input", "output"]
            print("Warning: No/invalid fields, use default: ['instruction', 'input', 'output'].")
        else:
            print(f"Using specified fields: {self.config['fields']}.")

        if "padding_side" not in self.config or self.config["padding_side"] not in ["left", "right"]:
            self.config["padding_side"] = "left"

        if "eps" not in self.config or not isinstance(self.config["eps"], (int, float)) or self.config["eps"] <= 0:
            self.config["eps"] = 1e-10

    def _setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = self.config["model"]

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                padding_side=self.config["padding_side"],
                trust_remote_code=bool(self.config.get("trust_remote_code", True)),
            )
            # Embedding models typically use fp16 + cuda; fp32 for cpu scenarios
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=bool(self.config.get("trust_remote_code", True)),
            )
        except Exception as e:
            print(f"Load specified embedding model failed ({e}), fall back to Qwen/Qwen3-Embedding-8B")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-Embedding-8B",
                padding_side=self.config["padding_side"],
                trust_remote_code=True,
            )
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModel.from_pretrained(
                "Qwen/Qwen3-Embedding-8B",
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )

        # Ensure tokenizer has pad_token
        if self.tokenizer.pad_token is None:
            # Embedding models also usually have eos_token; if not, use unk as a fallback
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        # If tokenizer has a known model_max_length, cap config max_length to avoid runtime errors.
        tok_max = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tok_max, int) and 0 < tok_max < 1_000_000:
            if int(self.config.get("max_length", tok_max)) > tok_max:
                print(
                    f"Warning: config max_length={self.config.get('max_length')} exceeds tokenizer.model_max_length={tok_max}. "
                    f"Capping max_length to {tok_max}."
                )
                self.config["max_length"] = tok_max

        self.model.to(self.device)
        self.model.eval()
        print("Setting up EmbedSVDEntropyScorer successfully")

    @staticmethod
    def _build_text(item: Dict, fields: List[str]) -> str:
        parts: List[str] = []
        for f in fields:
            v = item.get(f, "")
            if v is None:
                continue
            v = str(v).strip()
            if v:
                parts.append(v)
        return "\n".join(parts)

    def _compute_v_inf(self, last_hidden_state_2d: torch.Tensor) -> float:
        """
        last_hidden_state_2d: [L, H] (fp32)
        """
        if last_hidden_state_2d is None or last_hidden_state_2d.numel() == 0 or last_hidden_state_2d.dim() != 2:
            return 0.0

        try:
            singular_values = torch.linalg.svdvals(last_hidden_state_2d)
        except Exception as e:
            print(f"Warning: SVD failed ({e}), returning 0.0")
            return 0.0

        if singular_values.numel() == 0:
            return 0.0

        sum_sigma = torch.sum(singular_values)
        if not torch.isfinite(sum_sigma) or sum_sigma <= 0:
            return 0.0

        p_j = singular_values / sum_sigma
        eps = float(self.config.get("eps", 1e-10))
        v_inf = -torch.sum(p_j * torch.log(p_j + eps))

        if not torch.isfinite(v_inf):
            return 0.0
        return float(v_inf.item())

    def score_item(self, data_item):
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[float]:
        texts = [self._build_text(it, self.config["fields"]) for it in data_items]

        # Directly assign 0 to empty texts
        scores: List[float] = [0.0] * len(data_items)
        valid_indices = [i for i, t in enumerate(texts) if t]
        if not valid_indices:
            return scores

        valid_texts = [texts[i] for i in valid_indices]

        # Tokenize once (batch)
        enc = self.tokenizer(
            valid_texts,
            padding=True,
            truncation=True,
            max_length=self.config["max_length"],
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**enc)
            # [B, L, H]
            hidden = outputs.last_hidden_state

        attn = enc.get("attention_mask", None)  # [B, L]
        for j, original_idx in enumerate(valid_indices):
            # 1) Get the raw feature matrix (Sequence Length * Hidden Size)
            full_matrix = hidden[j]  # [L, H]

            # 2) Remove padding tokens: retain only rows where attention_mask==1
            if attn is not None:
                mask = attn[j].to(dtype=torch.bool)
                actual_matrix = full_matrix[mask]
            else:
                actual_matrix = full_matrix

            # If all tokens are masked out, score 0.
            if actual_matrix is None or actual_matrix.numel() == 0:
                scores[original_idx] = 0.0
                continue

            # 2.5) Cap token rows for SVD to prevent extreme compute / OOM on long sequences.
            # Strategy is controlled by svd_token_strategy:
            # - tail: keep the LAST svd_max_tokens tokens (after removing padding)
            # - uniform: uniformly sample svd_max_tokens positions across the sequence
            svd_max_tokens = int(self.config.get("svd_max_tokens", 512))
            if svd_max_tokens > 0 and actual_matrix.size(0) > svd_max_tokens:
                strategy = str(self.config.get("svd_token_strategy", "tail"))
                if strategy == "uniform":
                    idx = torch.linspace(
                        0,
                        actual_matrix.size(0) - 1,
                        steps=svd_max_tokens,
                        device=actual_matrix.device,
                    ).long()
                    actual_matrix = actual_matrix.index_select(0, idx)
                else:
                    # default: tail
                    actual_matrix = actual_matrix[-svd_max_tokens:]

            # 3) SVD requires fp32 precision
            actual_matrix_fp32 = actual_matrix.to(torch.float32)

            # 4) Eigenvalue decomposition/SVD + normalization + entropy
            scores[original_idx] = self._compute_v_inf(actual_matrix_fp32)

        return scores

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config.get("batch_size")
        buf_items, buf_ids = [], []

        with open(dataset, "r", encoding="utf-8", errors="ignore") as f:
            pbar = tqdm(total=num_lines, desc=self.config.get("name", "EmbedSVDEntropyScorer"))
            for line in f:
                line = line.strip()
                if not line:
                    pbar.update(1)
                    continue
                item = json.loads(line)
                buf_items.append(item)
                buf_ids.append(item.get("id", ""))

                if len(buf_items) == batch_size:
                    batch_scores = self.score_batch(buf_items)
                    results.extend({"id": _id, "score": sc} for _id, sc in zip(buf_ids, batch_scores))
                    buf_items.clear()
                    buf_ids.clear()
                pbar.update(1)

            if buf_items:
                batch_scores = self.score_batch(buf_items)
                results.extend({"id": _id, "score": sc} for _id, sc in zip(buf_ids, batch_scores))
                buf_items.clear()
                buf_ids.clear()
            pbar.close()

        return results

    def evaluate_to_file(self, dataset: str, output_file: str, resume: bool = True) -> str:
        """
        Stream pointwise results to JSONL (append), enabling resume from existing output_file.
        """
        num_lines = get_total_lines(dataset)
        batch_size = self.config.get("batch_size")

        done_ids = set()
        if resume and os.path.exists(output_file):
            repair_trailing_incomplete_jsonl(output_file)
            done_ids = load_jsonl_id_set(output_file, id_key="id")
            if done_ids:
                print(f"[EmbedSVDEntropyScorer] Resume enabled. Found {len(done_ids)} unique completed ids in {output_file}.")

        if not resume:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8"):
                pass

        buf_items, buf_ids = [], []

        with open(dataset, "r", encoding="utf-8", errors="ignore") as f:
            pbar = tqdm(total=num_lines, desc=self.config.get("name", "EmbedSVDEntropyScorer"))
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