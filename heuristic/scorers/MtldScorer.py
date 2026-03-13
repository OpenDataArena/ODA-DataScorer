import json
import string
import os
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from .base_scorer import BaseScorer
from .utils import get_total_lines
from utils.utils_jsonl import append_jsonl, repair_trailing_incomplete_jsonl, load_jsonl_id_set, normalize_id


# Helper function for multiprocessing (must be at module level for pickling)
def _process_single_line(args):
    """Helper function to process a single line (for multiprocessing)
    
    Args:
        args: Tuple of (line, ttr_threshold)
    
    Returns:
        Dict containing id and MTLD_Score
    """
    line, ttr_threshold = args
    
    try:
        item = json.loads(line.strip())
        
        # Extract text
        instruction = item["instruction"]
        input_text = item.get("input", "")
        response = item["output"]
        
        # Concatenate text
        if input_text:
            text = instruction + '\n' + input_text + '\n' + response
        else:
            text = instruction + '\n' + response
        
        # Calculate MTLD score
        score = _compute_mtld(text.split(), ttr_threshold)
        
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


def _mtld_calc(word_array, ttr_threshold, remove_punctuation):
    """Internal method to calculate MTLD"""
    current_ttr = 1.0
    token_count = 0
    type_count = 0
    types = set()
    factors = 0.0

    for token in word_array:
        # trim punctuation, make lowercase
        token = token.translate(remove_punctuation).lower()
        token_count += 1
        if token not in types:
            type_count += 1
            types.add(token)
        current_ttr = type_count / token_count
        if current_ttr <= ttr_threshold:
            factors += 1
            token_count = 0
            type_count = 0
            types = set()
            current_ttr = 1.0

    excess = 1.0 - current_ttr
    excess_val = 1.0 - ttr_threshold
    factors += excess / excess_val
    if factors != 0:
        return len(word_array) / factors
    return -1


def _compute_mtld(word_array, ttr_threshold=0.72):
    """
    Calculate MTLD (Measure of Textual Lexical Diversity) score
    Used to measure lexical diversity of text
    """
    if isinstance(word_array, str):
        raise ValueError(
            "Input should be a list of strings, rather than a string. Try using string.split()")
    if len(word_array) < 1:
        return 0.0  # Return 0 instead of raising exception
    
    # Set up translation table for removing punctuation
    remove_punctuation = str.maketrans('', '', string.punctuation)
    
    return (_mtld_calc(word_array, ttr_threshold, remove_punctuation) + 
            _mtld_calc(word_array[::-1], ttr_threshold, remove_punctuation)) / 2


class MtldScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        # TTR threshold parameter validation
        if "ttr_threshold" in self.config and isinstance(self.config["ttr_threshold"], (int, float)) and 0 < self.config["ttr_threshold"] < 1:
            print(f"Using specified ttr_threshold: {self.config['ttr_threshold']}.")
        elif "ttr_threshold" in self.config and isinstance(self.config["ttr_threshold"], (int, float)) and not (0 < self.config["ttr_threshold"] < 1):
            print("Warning: ttr_threshold should be between 0 and 1, using default value of 0.72.")
            self.config['ttr_threshold'] = 0.72
        else:
            print("Warning: No specific ttr_threshold, using default value of 0.72.")
            self.config['ttr_threshold'] = 0.72
        
        # Multiprocessing worker count validation
        if "max_workers" in self.config and isinstance(self.config["max_workers"], int) and self.config["max_workers"] > 0:
            print(f"Using specified max_workers: {self.config['max_workers']}.")
        else:
            import os
            # Default to CPU core count
            default_workers = max(1, os.cpu_count() or 1)
            print(f"Warning: No/invalid max_workers, using default value of {default_workers} (CPU count).")
            self.config['max_workers'] = default_workers

    def _setup(self):
        """Initialize MtldScorer"""
        print("Setting up MtldScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        """Score a single data item"""
        instruction = data_item["instruction"]
        input_text = data_item.get("input", "")
        response = data_item["output"]
        
        # Concatenate text
        if input_text:
            text = instruction + '\n' + input_text + '\n' + response
        else:
            text = instruction + '\n' + response
        
        # Use ttr_threshold from configuration
        ttr_threshold = self.config.get('ttr_threshold', 0.72)
        return _compute_mtld(text.split(), ttr_threshold=ttr_threshold)

    def evaluate(self, dataset) -> List[Dict]:
        """Evaluate the entire dataset"""
        num_lines = get_total_lines(dataset)
        max_workers = self.config.get('max_workers', 1)
        ttr_threshold = self.config.get('ttr_threshold', 0.72)
        
        print(f"Using {max_workers} worker(s) for parallel processing")
        
        # Read all lines and prepare tasks
        with open(dataset, 'r', encoding='utf-8') as f:
            lines = [line for line in f]
        
        # Prepare task parameters
        tasks = [(line, ttr_threshold) for line in lines]
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to display progress bar
            results = list(tqdm(
                executor.map(_process_single_line, tasks),
                total=num_lines,
                desc=self.config.get('name', 'MtldScorer')
            ))
        
        return results

    def evaluate_to_file(self, dataset: str, output_file: str, resume: bool = True) -> str:
        """
        Stream pointwise results to JSONL (append), enabling resume from existing output_file.
        Resume logic: load finished ids from output_file, skip them when reading dataset.
        """
        num_lines = get_total_lines(dataset)
        max_workers = self.config.get("max_workers", 1)
        ttr_threshold = self.config.get("ttr_threshold", 0.72)

        done_ids = set()
        if resume and os.path.exists(output_file):
            repair_trailing_incomplete_jsonl(output_file)
            done_ids = load_jsonl_id_set(output_file, id_key="id")
            if done_ids:
                print(f"[MtldScorer] Resume enabled. Found {len(done_ids)} unique completed ids in {output_file}.")

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
        pbar = tqdm(total=num_lines, desc=self.config.get("name", "MtldScorer"))

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

                    chunk_args.append((line, ttr_threshold))

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