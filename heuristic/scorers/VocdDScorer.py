import json
import os
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from lexicalrichness import LexicalRichness
from .base_scorer import BaseScorer
from .utils import get_total_lines
from utils.utils_jsonl import append_jsonl, repair_trailing_incomplete_jsonl, load_jsonl_id_set, normalize_id


# Helper function for multiprocessing (must be at module level for pickling)
def _vocd_score_helper(text: str, ntokens: int, within_sample: int, seed: int) -> float:
    """
    Calculate VOCD-D score for measuring vocabulary diversity of text
    
    Args:
        text: Text string to analyze
        ntokens: Maximum number of tokens for sampling
        within_sample: Number of samples for each size
        seed: Random seed
        
    Returns:
        VOCD-D score (float)
    """
    if not text or len(text.strip()) == 0:
        return 0.0
    
    try:
        # Initialize LexicalRichness object
        lex = LexicalRichness(text)
        
        # Check if text has enough vocabulary for sampling
        if lex.words < ntokens:  # VOCD requires sufficient vocabulary to support configured ntokens
            return 0.0
        
        # Calculate vocd-D value
        vocd_d_value = lex.vocd(
            ntokens=ntokens,
            within_sample=within_sample,
            seed=seed
        )
        
        return vocd_d_value
    except Exception as e:
        # Return 0 if calculation fails
        print(f"Error computing VOCD-D: {e}")
        return 0.0


def _process_single_line(args):
    """Helper function to process a single line (for multiprocessing)
    
    Args:
        args: Tuple of (line, ntokens, within_sample, seed)
    
    Returns:
        Dict containing id and VOCD_Score
    """
    line, ntokens, within_sample, seed = args
    
    try:
        item = json.loads(line.strip())
        
        instruction = item["instruction"]
        input_text = item.get("input", "")
        response = item["output"]
        
        # Concatenate text
        if input_text:
            text = instruction + '\n' + input_text + '\n' + response
        else:
            text = instruction + '\n' + response
        
        # Calculate VOCD score
        score = _vocd_score_helper(text, ntokens, within_sample, seed)
        
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


class VocdDScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate ntokens parameter: maximum number of tokens for sampling
        if "ntokens" in self.config and isinstance(self.config["ntokens"], int) and self.config["ntokens"] > 0:
            print(f"Using specified ntokens: {self.config['ntokens']}.")
        elif "ntokens" in self.config and isinstance(self.config["ntokens"], int) and self.config["ntokens"] <= 0:
            print("Warning: ntokens should be > 0, using default value of 50.")
            self.config['ntokens'] = 50
        else:
            print("Warning: No specific ntokens, using default value of 50.")
            self.config['ntokens'] = 50

        # Validate within_sample parameter: number of samples for each size
        if "within_sample" in self.config and isinstance(self.config["within_sample"], int) and self.config["within_sample"] > 0:
            print(f"Using specified within_sample: {self.config['within_sample']}.")
        elif "within_sample" in self.config and isinstance(self.config["within_sample"], int) and self.config["within_sample"] <= 0:
            print("Warning: within_sample should be > 0, using default value of 100.")
            self.config['within_sample'] = 100
        else:
            print("Warning: No specific within_sample, using default value of 100.")
            self.config['within_sample'] = 100

        # Validate seed parameter: random seed
        if "seed" in self.config and isinstance(self.config["seed"], int):
            print(f"Using specified seed: {self.config['seed']}.")
        else:
            print("Warning: No specific seed, using default value of 42.")
            self.config['seed'] = 42

        # Validate max_workers parameter: number of worker processes
        if "max_workers" in self.config and isinstance(self.config["max_workers"], int) and self.config["max_workers"] > 0:
            print(f"Using specified max_workers: {self.config['max_workers']}.")
        else:
            print("Warning: No/invalid max_workers, using default value of 128.")
            self.config['max_workers'] = 128

    def _setup(self):
        """Initialize VocdDScorer"""
        print("Setting up VocdDScorer successfully")

    def vocd_score(self, text: str) -> float:
        """
        Calculate VOCD-D score for measuring vocabulary diversity of text
        
        Args:
            text: Text string to analyze
            
        Returns:
            VOCD-D score (float)
        """
        ntokens = self.config.get('ntokens', 50)
        within_sample = self.config.get('within_sample', 100)
        seed = self.config.get('seed', 42)
        
        return _vocd_score_helper(text, ntokens, within_sample, seed)

    def score_item(self, data_item: Dict) -> float:
        """Calculate VOCD score for a single data item"""
        instruction = data_item["instruction"]
        input_text = data_item.get("input", "")
        response = data_item["output"]
        
        # Concatenate text
        if input_text:
            text = instruction + '\n' + input_text + '\n' + response
        else:
            text = instruction + '\n' + response
        
        return self.vocd_score(text)

    def evaluate(self, dataset) -> List[Dict]:
        """Evaluate the entire dataset"""
        num_lines = get_total_lines(dataset)
        max_workers = self.config.get('max_workers', 128)
        ntokens = self.config.get('ntokens', 50)
        within_sample = self.config.get('within_sample', 100)
        seed = self.config.get('seed', 42)
        
        print(f"Using {max_workers} worker(s) for parallel processing")
        
        # Read all lines and prepare tasks
        with open(dataset, 'r', encoding='utf-8') as f:
            lines = [line for line in f]
        
        # Prepare task parameters
        tasks = [(line, ntokens, within_sample, seed) for line in lines]
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to display progress bar
            results = list(tqdm(
                executor.map(_process_single_line, tasks),
                total=num_lines,
                desc=self.config.get('name', 'VocdDScorer')
            ))
        
        return results

    def evaluate_to_file(self, dataset: str, output_file: str, resume: bool = True) -> str:
        """
        Stream pointwise results to JSONL (append), enabling resume from existing output_file.
        Resume logic: load finished ids from output_file, skip them when reading dataset.
        """
        num_lines = get_total_lines(dataset)
        max_workers = self.config.get("max_workers", 128)
        ntokens = self.config.get("ntokens", 50)
        within_sample = self.config.get("within_sample", 100)
        seed = self.config.get("seed", 42)

        done_ids = set()
        if resume and os.path.exists(output_file):
            repair_trailing_incomplete_jsonl(output_file)
            done_ids = load_jsonl_id_set(output_file, id_key="id")
            if done_ids:
                print(f"[VocdDScorer] Resume enabled. Found {len(done_ids)} unique completed ids in {output_file}.")

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
        pbar = tqdm(total=num_lines, desc=self.config.get("name", "VocdDScorer"))

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
                        # Let worker handle malformed lines; cannot dedup by id here.
                        item_id = ""

                    if item_id and item_id in done_ids:
                        pbar.update(1)
                        continue

                    chunk_args.append((line, ntokens, within_sample, seed))

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

            # Process the remaining unfinished chunk
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
