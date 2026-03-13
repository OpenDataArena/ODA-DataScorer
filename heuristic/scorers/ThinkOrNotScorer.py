import json
import os
import re
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from .base_scorer import BaseScorer
from .utils import get_total_lines
from utils.utils_jsonl import append_jsonl, repair_trailing_incomplete_jsonl, load_jsonl_id_set, normalize_id


# Compile regex patterns for think tags (module level for multiprocessing)
THINK_PATTERNS = [
    re.compile(r'<think\s*>', re.IGNORECASE),                      # think opening tag (with optional spaces)
    re.compile(r'</think\s*>', re.IGNORECASE),                     # think closing tag (with optional spaces)
    re.compile(r'<redacted_reasoning\s*>', re.IGNORECASE),         # redacted_reasoning opening tag
    re.compile(r'</redacted_reasoning\s*>', re.IGNORECASE),        # redacted_reasoning closing tag
]


def _contains_think_tag(text: str) -> bool:
    """
    Check if text contains think tags
    
    Args:
        text: Text to check
        
    Returns:
        bool: True if contains think tags, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
    
    # Check all patterns
    for pattern in THINK_PATTERNS:
        if pattern.search(text):
            return True
    
    return False


# Helper function for multiprocessing (must be at module level for pickling)
def _process_single_line(args):
    """Helper function to process a single line (for multiprocessing)
    
    Args:
        args: Tuple of (line, field_name)
    
    Returns:
        Dict containing id and ThinkOrNot_Score
    """
    line, field_name = args
    
    try:
        item = json.loads(line.strip())
        
        # Extract text from specified field
        text = item.get(field_name, "")
        
        # Check if contains think tags
        if _contains_think_tag(text):
            score = 1.0
        else:
            score = 0.0
        
        return {
            "id": item.get("id", ""),
            "score": score
        }
    except Exception as e:
        # If processing fails, return a result with error marker
        return {
            "id": item.get("id", "unknown") if 'item' in locals() else "unknown",
            "score": 0.0,
            "error": str(e)
        }


class ThinkOrNotScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        # Check the field to examine
        if "field" not in self.config or not isinstance(self.config["field"], str):
            self.config["field"] = "output"
            print("Warning: No/invalid field specified, use default value of 'output'.")
        else:
            print(f"Using specified field: {self.config['field']}.")

        # Multiprocessing worker count validation
        if "max_workers" in self.config and isinstance(self.config["max_workers"], int) and self.config["max_workers"] > 0:
            print(f"Using specified max_workers: {self.config['max_workers']}.")
        else:
            # Default to CPU core count
            default_workers = max(1, os.cpu_count() or 1)
            print(f"Warning: No/invalid max_workers, using default value of {default_workers} (CPU count).")
            self.config['max_workers'] = default_workers

    def _setup(self):
        """Initialize ThinkOrNotScorer"""
        print("Setting up ThinkOrNotScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        """
        Score a single data item
        Check if data contains LLM's thinking part
        
        Args:
            data_item: Data item dictionary
            
        Returns:
            float: 1.0 if contains think tags, 0.0 otherwise
        """
        # Extract text from specified field
        field_name = self.config["field"]
        text = data_item.get(field_name, "")
        
        # If field doesn't exist or is empty, return 0.0
        if not text or not isinstance(text, str):
            return 0.0
        
        # Check if contains think tags
        if _contains_think_tag(text):
            return 1.0
        else:
            return 0.0

    def evaluate(self, dataset) -> List[Dict]:
        """Evaluate the entire dataset"""
        num_lines = get_total_lines(dataset)
        max_workers = self.config.get('max_workers', 1)
        field_name = self.config.get('field', 'output')
        
        print(f"Using {max_workers} worker(s) for parallel processing")
        
        # Read all lines and prepare tasks
        with open(dataset, 'r', encoding='utf-8') as f:
            lines = [line for line in f]
        
        # Prepare task parameters
        tasks = [(line, field_name) for line in lines]
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to display progress bar
            results = list(tqdm(
                executor.map(_process_single_line, tasks),
                total=num_lines,
                desc=self.config.get('name', 'ThinkOrNotScorer')
            ))
        
        return results

    def evaluate_to_file(self, dataset: str, output_file: str, resume: bool = True) -> str:
        """
        Stream pointwise results to JSONL (append), enabling resume from existing output_file.
        Resume logic: load finished ids from output_file, skip them when reading dataset.
        """
        num_lines = get_total_lines(dataset)
        max_workers = self.config.get("max_workers", 1)
        field_name = self.config.get("field", "output")

        done_ids = set()
        if resume and os.path.exists(output_file):
            repair_trailing_incomplete_jsonl(output_file)
            done_ids = load_jsonl_id_set(output_file, id_key="id")
            if done_ids:
                print(f"[ThinkOrNotScorer] Resume enabled. Found {len(done_ids)} unique completed ids in {output_file}.")

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
        pbar = tqdm(total=num_lines, desc=self.config.get("name", "ThinkOrNotScorer"))

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

                    chunk_args.append((line, field_name))

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
