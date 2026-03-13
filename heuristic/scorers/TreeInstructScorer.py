import json
import spacy
import os
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from .base_scorer import BaseScorer
from .utils import get_total_lines
from utils.utils_jsonl import append_jsonl, repair_trailing_incomplete_jsonl, load_jsonl_id_set, normalize_id


# Global cache for spacy models to avoid reloading in each process
_spacy_model_cache = {}


def _get_spacy_model(model_name: str):
    """Get or load spacy model from cache
    
    Args:
        model_name: Name of the spacy model to load
        
    Returns:
        Loaded spacy model
    """
    global _spacy_model_cache
    if model_name not in _spacy_model_cache:
        try:
            _spacy_model_cache[model_name] = spacy.load(model_name)
        except OSError:
            print(f"Error: Spacy model '{model_name}' not found. Please install it using:")
            print(f"  python -m spacy download {model_name}")
            raise
    return _spacy_model_cache[model_name]


def _count_nodes(token):
    """
    Recursively count the number of nodes in the subtree rooted at token.
    
    Args:
        token: Root token of the subtree
        
    Returns:
        Number of nodes in the subtree
    """
    return 1 + sum(_count_nodes(child) for child in token.children)


def _tree_depth(token):
    """
    Recursively calculate the depth (number of levels) of the subtree rooted at token.
    A single word has depth 1.
    
    Args:
        token: Root token of the subtree
        
    Returns:
        Depth of the subtree
    """
    children = list(token.children)
    if not children:
        return 1
    return 1 + max(_tree_depth(child) for child in children)


def _analyze_text(text: str, model_name: str) -> Dict[str, float]:
    """
    Analyze the syntactic tree structure of text, return node count and depth.
    
    Args:
        text: Text to analyze
        model_name: Name of the spacy model to use
        
    Returns:
        Dictionary containing total_nodes and max_depth
    """
    if not text or len(text.strip()) == 0:
        return {"total_nodes": 0, "max_depth": 0}
    
    nlp = _get_spacy_model(model_name)
    doc = nlp(text)
    
    # Find sentence roots (ROOT), usually only one
    roots = [token for token in doc if token.head == token]
    
    total_nodes = 0
    max_depth = 0
    
    for root in roots:
        subtree_nodes = _count_nodes(root)
        subtree_depth = _tree_depth(root)
        total_nodes += subtree_nodes
        max_depth = max(max_depth, subtree_depth)
    
    return {
        "total_nodes": total_nodes,
        "max_depth": max_depth
    }


# Helper function for multiprocessing (must be at module level for pickling)
def _process_single_line(args):
    """Helper function to process a single line (for multiprocessing)
    
    Args:
        args: Tuple of (line, model_name, text_fields)
    
    Returns:
        Dict containing id, TreeInstruct_Nodes, and TreeInstruct_Depth
    """
    line, model_name, text_fields = args
    
    try:
        item = json.loads(line.strip())
        
        # Concatenate text from specified fields
        text_parts = []
        for field in text_fields:
            if field in item and item[field]:
                text_parts.append(str(item[field]))
        
        text = '\n'.join(text_parts)
        
        # Analyze text
        result = _analyze_text(text, model_name)
        
        return {
            "id": item.get("id", ""),
            "TreeInstruct_Nodes": result["total_nodes"],
            "TreeInstruct_Depth": result["max_depth"]
        }
    except Exception as e:
        # If processing fails, return a result with error marker
        return {
            "id": item.get("id", "unknown") if 'item' in locals() else "unknown",
            "TreeInstruct_Nodes": 0,
            "TreeInstruct_Depth": 0,
            "error": str(e)
        }


class TreeInstructScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate spacy model name
        if "model" in self.config and isinstance(self.config["model"], str):
            print(f"Using specified spacy_model: {self.config['model']}.")
        else:
            print("Warning: No specific spacy_model, using default value 'en_core_web_sm'.")
            self.config['model'] = 'en_core_web_sm'
        
        # Validate max_workers (process count)
        if "max_workers" in self.config and isinstance(self.config["max_workers"], int) and self.config["max_workers"] > 0:
            print(f"Using specified max_workers: {self.config['max_workers']}.")
        else:
            import os
            # Default to CPU core count
            default_workers = max(1, os.cpu_count() or 1)
            print(f"Warning: No/invalid max_workers, using default value of {default_workers} (CPU count).")
            self.config['max_workers'] = default_workers
        
        # Validate text fields selection
        if "text_fields" in self.config and isinstance(self.config["text_fields"], list):
            print(f"Using specified text_fields: {self.config['text_fields']}.")
        else:
            print("Warning: No specific text_fields, using default ['instruction', 'input', 'output'].")
            self.config['text_fields'] = ['instruction', 'input', 'output']

    def _setup(self):
        """Initialize TreeInstructScorer"""
        # Validate spacy model by loading it once
        model = self.config.get('model', 'en_core_web_sm')
        try:
            spacy.load(model)
            print(f"Setting up TreeInstructScorer with spacy model '{model}' successfully")
        except OSError:
            print(f"Error: Spacy model '{model}' not found. Please install it using:")
            print(f"  python -m spacy download {model}")
            raise

    def score_item(self, data_item: Dict) -> Dict[str, float]:
        """Score a single data item"""
        text_fields = self.config.get('text_fields', ['instruction', 'input', 'output'])
        model_name = self.config.get('model', 'en_core_web_sm')
        
        # Concatenate text from specified fields
        text_parts = []
        for field in text_fields:
            if field in data_item and data_item[field]:
                text_parts.append(str(data_item[field]))
        
        text = '\n'.join(text_parts)
        
        # Analyze text
        result = _analyze_text(text, model_name)
        
        return result

    def evaluate(self, dataset) -> List[Dict]:
        """Evaluate the entire dataset"""
        num_lines = get_total_lines(dataset)
        max_workers = self.config.get('max_workers', 1)
        model_name = self.config.get('model', 'en_core_web_sm')
        text_fields = self.config.get('text_fields', ['instruction', 'input', 'output'])
        
        print(f"Using {max_workers} worker(s) for parallel processing")
        
        # Read all lines and prepare tasks
        with open(dataset, 'r', encoding='utf-8') as f:
            lines = [line for line in f]
        
        # Prepare task parameters
        tasks = [(line, model_name, text_fields) for line in lines]
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to display progress bar
            results = list(tqdm(
                executor.map(_process_single_line, tasks),
                total=num_lines,
                desc=self.config.get('name', 'TreeInstructScorer')
            ))
        
        return results

    def evaluate_to_file(self, dataset: str, output_file: str, resume: bool = True) -> str:
        """
        Stream pointwise results to JSONL (append), enabling resume from existing output_file.
        Resume logic: load finished ids from output_file, skip them when reading dataset.
        """
        num_lines = get_total_lines(dataset)
        max_workers = self.config.get("max_workers", 1)
        model_name = self.config.get("model", "en_core_web_sm")
        text_fields = self.config.get("text_fields", ['instruction', 'input', 'output'])

        done_ids = set()
        if resume and os.path.exists(output_file):
            repair_trailing_incomplete_jsonl(output_file)
            done_ids = load_jsonl_id_set(output_file, id_key="id")
            if done_ids:
                print(f"[TreeInstructScorer] Resume enabled. Found {len(done_ids)} unique completed ids in {output_file}.")

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
        pbar = tqdm(total=num_lines, desc=self.config.get("name", "TreeInstructScorer"))

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

                    chunk_args.append((line, model_name, text_fields))

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