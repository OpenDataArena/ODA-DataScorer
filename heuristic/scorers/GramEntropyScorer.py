from .base_scorer import BaseScorer
import json
import os
from typing import Dict, List
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from .utils import get_total_lines
from concurrent.futures import ProcessPoolExecutor
import math
from collections import Counter
from utils.utils_jsonl import append_jsonl, repair_trailing_incomplete_jsonl, load_jsonl_id_set, normalize_id


# Helper function for multiprocessing (must be at module level for pickling)
def _process_single_line(args):
    """Helper function to process a single line (for multiprocessing)
    
    Args:
        args: Tuple of (line, download_nltk)
    
    Returns:
        Dict containing id and score
    """
    line, download_nltk = args
    
    try:
        item = json.loads(line.strip())
        
        # Get text content from data item
        instruction = item["instruction"]
        input_text = item.get("input", "")
        response = item["output"]
        
        if input_text:
            text = instruction + '\n' + input_text + '\n' + response
        else:
            text = instruction + '\n' + response
        
        # Download NLTK data if needed (for this process)
        if download_nltk:
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                nltk.download('punkt_tab', quiet=True)
        
        text = text.lower()
        
        try:
            tokens = word_tokenize(text)
        except Exception as e:
            return {
                "id": item.get("id", ""),
                "score": 0.0,
                "error": f"Tokenization error: {str(e)}"
            }
        
        if len(tokens) == 0:
            # If no tokens, return 0
            return {
                "id": item.get("id", ""),
                "score": 0.0
            }
        
        # Count 1-gram frequencies
        unigram_counts = Counter(tokens)
        total_count = len(tokens)
        
        # Calculate entropy: H(X) = -Σ p(x) * log2(p(x))
        entropy = 0.0
        for count in unigram_counts.values():
            probability = count / total_count
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return {
            "id": item.get("id", ""),
            "score": entropy
        }
        
    except Exception as e:
        # If processing fails, return a result with error marker
        return {
            "id": item.get("id", "unknown") if 'item' in locals() else "unknown",
            "score": 0.0,
            "error": str(e)
        }


class GramEntropyScorer(BaseScorer):
    def _validate_config(self):
        # Check max_workers (process count)
        if "max_workers" not in self.config or not isinstance(self.config["max_workers"], int) or self.config["max_workers"] <= 0:
            # Default to CPU core count
            default_workers = max(1, os.cpu_count() or 1)
            print(f"Warning: No/invalid max_workers, using default value of {default_workers} (CPU count).")
            self.config['max_workers'] = default_workers
        else:
            print(f"Using specified max_workers: {self.config['max_workers']}.")

    def _setup(self):
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print("Downloading NLTK punkt_tab tokenizer...")
            nltk.download('punkt_tab', quiet=True)
            print("NLTK punkt_tab tokenizer downloaded successfully.")
        
        print("Setting up GramEntropyScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        """Calculate the 1-gram entropy value for a single data item"""
        instruction = data_item["instruction"]
        input_text = data_item.get("input", "")
        response = data_item["output"]
        
        if input_text:
            text = instruction + '\n' + input_text + '\n' + response
        else:
            text = instruction + '\n' + response
        
        text = text.lower()
        tokens = word_tokenize(text)
        
        if len(tokens) == 0:
            # If no tokens, return 0
            return 0.0
        
        # Count 1-gram frequencies
        unigram_counts = Counter(tokens)
        total_count = len(tokens)
        
        # Calculate entropy: H(X) = -Σ p(x) * log2(p(x))
        entropy = 0.0
        for count in unigram_counts.values():
            probability = count / total_count
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy

    def evaluate(self, dataset) -> List[Dict]:
        """Evaluate the entire dataset"""
        num_lines = get_total_lines(dataset)
        max_workers = self.config.get('max_workers', 1)
        
        print(f"Using {max_workers} worker(s) for parallel processing")
        
        # Read all lines and prepare tasks
        with open(dataset, 'r', encoding='utf-8') as f:
            lines = [line for line in f]
        
        # Prepare task parameters
        tasks = [(line, True) for line in lines]
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to display progress bar
            results = list(tqdm(
                executor.map(_process_single_line, tasks),
                total=num_lines,
                desc=self.config.get('name', 'GramEntropyScorer')
            ))
        
        return results

    def evaluate_to_file(self, dataset: str, output_file: str, resume: bool = True) -> str:
        """
        Stream pointwise results to JSONL (append), enabling resume from existing output_file.
        Resume logic: load finished ids from output_file, skip them when reading dataset.
        """
        num_lines = get_total_lines(dataset)
        max_workers = self.config.get("max_workers", 1)
        download_nltk = True

        done_ids = set()
        if resume and os.path.exists(output_file):
            repair_trailing_incomplete_jsonl(output_file)
            done_ids = load_jsonl_id_set(output_file, id_key="id")
            if done_ids:
                print(f"[GramEntropyScorer] Resume enabled. Found {len(done_ids)} unique completed ids in {output_file}.")

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
        pbar = tqdm(total=num_lines, desc=self.config.get("name", "GramEntropyScorer"))

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

                    chunk_args.append((line, download_nltk))

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

