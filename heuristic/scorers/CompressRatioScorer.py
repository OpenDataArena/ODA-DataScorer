import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

from tqdm import tqdm

from .base_scorer import BaseScorer
from .utils import get_total_lines
from utils.utils_jsonl import (
    append_jsonl,
    load_jsonl_id_set,
    normalize_id,
    repair_trailing_incomplete_jsonl,
)


def _calc_compression_ratio(text: str, level: int = 9) -> Dict:
    """
    计算压缩比：compressed_size / original_size
    - original_size: 原始字节数
    - compressed_size: zlib(level) 压缩后的字节数
    - ratio: 压缩比（越小越“可压缩/重复度高”）
    """
    if not text:
        return {"original_size": 0, "compressed_size": 0, "ratio": 0.0}

    original_bytes = text.encode("utf-8", errors="ignore")
    original_size = len(original_bytes)
    if original_size == 0:
        return {"original_size": 0, "compressed_size": 0, "ratio": 0.0}

    compressed_bytes = zlib.compress(original_bytes, level=level)
    compressed_size = len(compressed_bytes)
    ratio = compressed_size / original_size
    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "ratio": round(float(ratio), 4),
    }


# Helper function for multiprocessing (must be at module level for pickling)
def _process_single_line(args):
    """Helper function to process a single line (for multiprocessing)

    Args:
        args: Tuple of (line, fields, level)

    Returns:
        Dict containing id and score (ratio). Optionally includes sizes.
    """
    line, fields, level = args

    try:
        item = json.loads(line.strip())

        parts = []
        for field in fields:
            if field in item and item[field]:
                parts.append(str(item[field]))
        text = "\n".join(parts) if parts else ""

        metrics = _calc_compression_ratio(text, level=level)
        return {"id": item.get("id", ""), "score": metrics["ratio"]}
    except Exception as e:
        return {
            "id": item.get("id", "unknown") if "item" in locals() else "unknown",
            "score": 0.0,
            "error": str(e),
        }


class CompressRatioScorer(BaseScorer):
    def _validate_config(self):
        # fields: 默认与 TokenLengthScorer 一致
        if (
            "fields" not in self.config
            or not isinstance(self.config["fields"], list)
            or len(self.config["fields"]) == 0
        ):
            print(
                "Warning: No fields specified in config. Using default fields: ['instruction', 'input', 'output']."
            )
            self.config["fields"] = ["instruction", "input", "output"]
        else:
            print(f"Using specified fields: {self.config['fields']}.")

        # zlib compression level: 0~9
        level = self.config.get("level", 9)
        if not isinstance(level, int) or not (0 <= level <= 9):
            print("Warning: level should be int in [0, 9], using default value 9.")
            level = 9
        self.config["level"] = level

        # max_workers
        if (
            "max_workers" not in self.config
            or not isinstance(self.config["max_workers"], int)
            or self.config["max_workers"] <= 0
        ):
            default_workers = max(1, os.cpu_count() or 1)
            print(
                f"Warning: No/invalid max_workers, using default value of {default_workers} (CPU count)."
            )
            self.config["max_workers"] = default_workers
        else:
            print(f"Using specified max_workers: {self.config['max_workers']}.")

    def _setup(self):
        print("Setting up CompressRatioScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        fields = self.config["fields"]
        parts = []
        for field in fields:
            if field in data_item and data_item[field]:
                parts.append(str(data_item[field]))
        text = "\n".join(parts) if parts else ""
        metrics = _calc_compression_ratio(text, level=self.config.get("level", 9))
        return metrics["ratio"]

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        max_workers = self.config.get("max_workers", 1)
        fields = self.config.get("fields", ["instruction", "input", "output"])
        level = self.config.get("level", 9)

        print(f"Using {max_workers} worker(s) for parallel processing")

        with open(dataset, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line for line in f]

        tasks = [(line, fields, level) for line in lines]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(_process_single_line, tasks),
                    total=num_lines,
                    desc=self.config.get("name", "CompressRatioScorer"),
                )
            )

        return results

    def evaluate_to_file(self, dataset: str, output_file: str, resume: bool = True) -> str:
        """
        Stream pointwise results to JSONL (append), enabling resume from existing output_file.
        Resume logic: load finished ids from output_file, skip them when reading dataset.
        """
        num_lines = get_total_lines(dataset)
        max_workers = self.config.get("max_workers", 1)
        fields = self.config.get("fields", ["instruction", "input", "output"])
        level = self.config.get("level", 9)

        done_ids = set()
        if resume and os.path.exists(output_file):
            repair_trailing_incomplete_jsonl(output_file)
            done_ids = load_jsonl_id_set(output_file, id_key="id")
            if done_ids:
                print(
                    f"[CompressRatioScorer] Resume enabled. Found {len(done_ids)} unique completed ids in {output_file}."
                )

        if not resume:
            out_dir = os.path.dirname(output_file)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(output_file, "w", encoding="utf-8"):
                pass

        chunk_size = self.config.get("chunk_size", 2000)
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            chunk_size = 2000

        print(f"Using {max_workers} worker(s) for parallel processing")
        pbar = tqdm(total=num_lines, desc=self.config.get("name", "CompressRatioScorer"))

        buf_records = []
        chunk_args = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            with open(dataset, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line_stripped = line.strip()
                    if not line_stripped:
                        pbar.update(1)
                        continue

                    item_id = ""
                    try:
                        obj = json.loads(line_stripped)
                        item_id = normalize_id(obj.get("id", ""))
                    except Exception:
                        item_id = ""

                    if item_id and item_id in done_ids:
                        pbar.update(1)
                        continue

                    chunk_args.append((line, fields, level))

                    if len(chunk_args) >= chunk_size:
                        for rec in executor.map(_process_single_line, chunk_args):
                            buf_records.append(rec)
                            rid = normalize_id(rec.get("id", ""))
                            if rid:
                                done_ids.add(rid)

                            if len(buf_records) >= 1000:
                                append_jsonl(buf_records, output_file, flush=True)
                                buf_records.clear()

                            pbar.update(1)
                        chunk_args.clear()

            if chunk_args:
                for rec in executor.map(_process_single_line, chunk_args):
                    buf_records.append(rec)
                    rid = normalize_id(rec.get("id", ""))
                    if rid:
                        done_ids.add(rid)

                    if len(buf_records) >= 1000:
                        append_jsonl(buf_records, output_file, flush=True)
                        buf_records.clear()

                    pbar.update(1)
                chunk_args.clear()

        if buf_records:
            append_jsonl(buf_records, output_file, flush=True)

        pbar.close()
        return output_file