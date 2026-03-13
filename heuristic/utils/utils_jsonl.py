from typing import List
import json
from tqdm import tqdm
import os
import sys
from typing import Optional, Tuple


def save_jsonl(data, path):
    if "/" in path:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        # If it's a dictionary (global scores), save the entire dictionary as one line
        if isinstance(data, dict):
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        # If it's a list (scores for each sample), save line by line
        else:
            for sample in tqdm(data):
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def append_jsonl(records, path: str, flush: bool = True):
    """
    Append a list/iterable of JSON-serializable records to a JSONL file.
    """
    if "/" in path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if flush:
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass


def count_valid_jsonl_lines(path: str) -> int:
    """
    Count valid JSON lines in a JSONL file (ignores blank lines and malformed JSON lines).
    This is designed for resume/skip logic without loading the whole file into memory.
    """
    if not os.path.exists(path):
        return 0
    cnt = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                cnt += 1
            except Exception:
                # ignore malformed trailing line
                continue
    return cnt


def repair_trailing_incomplete_jsonl(path: str) -> bool:
    """
    If a crash left an incomplete trailing JSON line, truncate it.
    Returns True if file was modified.
    """
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            data = f.read()
        if not data:
            return False
        # If already ends with newline, assume no partial write.
        if data.endswith(b"\n"):
            return False

        # Find last newline; truncate to it. If none, truncate to empty.
        last_nl = data.rfind(b"\n")
        new_data = data[: last_nl + 1] if last_nl != -1 else b""

        with open(path, "wb") as f:
            f.write(new_data)
        return True
    except Exception:
        return False


def normalize_id(value) -> str:
    """
    Normalize an id value to a stable string for dedup/resume matching.
    This makes resume robust across int/str differences.
    """
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def load_jsonl_id_set(path: str, id_key: str = "id") -> set:
    """
    Load all ids from a pointwise JSONL output file into a set (deduplicated).
    Ignores blank/malformed lines and lines missing id_key.
    """
    ids = set()
    if not os.path.exists(path):
        return ids
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            if id_key not in obj:
                continue
            ids.add(normalize_id(obj.get(id_key)))
    return ids


def load_jsonl(path, max_lines=None):
    data = []
    skip_count = 0

    # Improve integer-to-string conversion limits (Python 3.11+)
    sys.set_int_max_str_digits(0)

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(tqdm(f), start=1):
                if max_lines is not None and i > max_lines:
                    break
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"[Warning] JSON decode error at line {i}: {e}")
                    skip_count += 1
                    continue
                except Exception as e:
                    print(f"[Warning] Unexpected error at line {i}: {e}")
                    skip_count += 1
                    continue
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while opening/reading the file: {e}")
        return []

    print(
        f"Total lines processed: {len(data) + skip_count}, Successfully loaded: {len(data)}, Skipped: {skip_count}"
    )
    return data


def merge_multiple_scores(
    file_a_path: str, b_file_paths: List[str], output_path: str, verbose=False
):
    """
    Merge multiple JSONL score files (B) into the main JSONL file (A), updating fields in Q_scores or QA_scores.

    :param file_a_path: Path to A file (main data)
    :param b_file_paths: List of B file paths (each containing id + one or more score fields)
    :param output_path: Path to the merged output JSONL file
    :param verbose: If True, prints warnings for unmatched score fields
    """

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Step 1: Load all score data from B files
    print("Step 1: Loading scores from all scorers' output files...")
    id_to_scores = {}
    global_scores = {}  # Store global scores (e.g., VendiScorer results)

    for b_path in b_file_paths:
        print(f"  Reading {b_path}")
        with open(b_path, "r", encoding="utf-8") as fb:
            lines = fb.readlines()
            
            # Check if it's a global score file (only one line and no id field)
            if len(lines) == 1:
                data = json.loads(lines[0])
                if "id" not in data:
                    # This is a global score, save to global_scores
                    scorer_name = os.path.basename(b_path).replace(".jsonl", "")
                    global_scores[scorer_name] = data
                    print(f"  Detected global score from {scorer_name}: {data}")
                    continue
            
            # Process normal per-sample scores
            for line in lines:
                data = json.loads(line)
                entry_id = data.get("id")
                if entry_id is None:
                    continue
                for key, value in data.items():
                    if key != "id":
                        id_to_scores.setdefault(entry_id, {})[key] = value

    print(f"Collected scores for {len(id_to_scores)} unique IDs.")
    if global_scores:
        print(f"Collected {len(global_scores)} global score(s): {list(global_scores.keys())}")

    # Step 2: Read A and insert scores
    print("Step 2: Processing original input file and inserting scores...")

    total = 0
    updated = 0

    with open(file_a_path, "r", encoding="utf-8") as fa, open(
        output_path, "w", encoding="utf-8"
    ) as fout, tqdm(desc="Progress", unit=" lines") as pbar:

        for line in fa:
            total += 1
            data = json.loads(line)
            entry_id = data.get("id")

            # Insert per-sample scores
            if entry_id in id_to_scores:
                for score_key, score_value in id_to_scores[entry_id].items():
                    inserted = False
                    if score_key in data.get("Q_scores", {}):
                        data["Q_scores"][score_key] = score_value
                        inserted = True
                    elif score_key in data.get("QA_scores", {}):
                        data["QA_scores"][score_key] = score_value
                        inserted = True
                    elif verbose:
                        print(
                            f"[Warning] id={entry_id} — field '{score_key}' not found in Q_scores or QA_scores"
                        )

                    if inserted:
                        updated += 1
            
            # Insert global scores (add the same global scores to every sample)
            if global_scores:
                if "global_scores" not in data:
                    data["global_scores"] = {}
                data["global_scores"].update(global_scores)

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            pbar.update(1)
    
    print(f"Total lines: {total}, Updated score fields: {updated}")
    if global_scores:
        print(f"Added global scores to all samples: {list(global_scores.keys())}")


def merge_jsonl_files(input_paths, output_path):
    """
    Merge multiple JSONL files into a single JSONL file.

    Args:
        input_paths (List[str]): List of paths to the .jsonl files to be merged.
        output_path (str): Path to the output .jsonl file after merging.
    """
    with open(output_path, "w", encoding="utf-8") as fout:
        for path in input_paths:
            with open(path, "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)


def add_id_to_jsonl(input_path: str, output_path: str):

    ds = load_jsonl(input_path)
    ds_add_id = []

    for idx, item in enumerate(ds):
        if 'id' in item and item['id'] is not None:
            pass
        else:
            item['id']=idx
        
        assert "instruction" in item and "output" in item, f"item {idx} is not valid"
        
        ds_add_id.append(item)
    save_jsonl(ds_add_id, output_path)
