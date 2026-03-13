import json
import os
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from .base_scorer import BaseScorer
from .utils import get_total_lines
from utils.utils_jsonl import append_jsonl, repair_trailing_incomplete_jsonl, load_jsonl_id_set, normalize_id


# -----------------------------
# Multiprocessing helpers
# -----------------------------
_GLOBAL_FINE_WORDS_LOWER: List[str] = []
_GLOBAL_RETURN_COUNTS: bool = False
_GLOBAL_ORIG_FINE_WORDS: List[str] = []
_GLOBAL_FIELDS: List[str] = []
_GLOBAL_MATCH_MODE: str = "substring"
_GLOBAL_PUNCT_TRANSLATION = None
_GLOBAL_FINE_WORDS_SET = None


def _init_worker(
    fine_words_lower: List[str],
    return_counts: bool,
    orig_fine_words: Optional[List[str]] = None,
    fields: Optional[List[str]] = None,
    match_mode: str = "substring",
):
    """
    Initializer for ProcessPoolExecutor workers.
    Avoids pickling fine_words on every task submission (more efficient than passing in args).
    """
    global _GLOBAL_FINE_WORDS_LOWER, _GLOBAL_RETURN_COUNTS, _GLOBAL_ORIG_FINE_WORDS, _GLOBAL_FIELDS
    global _GLOBAL_MATCH_MODE, _GLOBAL_PUNCT_TRANSLATION, _GLOBAL_FINE_WORDS_SET
    _GLOBAL_FINE_WORDS_LOWER = fine_words_lower or []
    _GLOBAL_RETURN_COUNTS = bool(return_counts)
    _GLOBAL_ORIG_FINE_WORDS = orig_fine_words or []
    _GLOBAL_FIELDS = fields or []
    _GLOBAL_MATCH_MODE = match_mode or "substring"

    # Prepare fast tokenization helpers for token-mode
    if _GLOBAL_MATCH_MODE == "token":
        # Replace punctuation with spaces, then split on whitespace.
        # Include ascii punct and common CJK punct to be robust.
        punct = (
            r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
            + "，。、；：？！“”‘’（）【】《》…—·"
        )
        _GLOBAL_PUNCT_TRANSLATION = str.maketrans({ch: " " for ch in punct})
        _GLOBAL_FINE_WORDS_SET = set([w for w in _GLOBAL_FINE_WORDS_LOWER if w])
    else:
        _GLOBAL_PUNCT_TRANSLATION = None
        _GLOBAL_FINE_WORDS_SET = None


def _build_text_from_item(item: Dict[str, Any], fields: List[str]) -> str:
    """
    Build text by concatenating specified fields, consistent with TokenLengthScorer.
    Empty/missing fields are skipped. Joined by newline.
    """
    parts: List[str] = []
    for field in fields:
        if field in item and item[field]:
            parts.append(str(item[field]))
    return "\n".join(parts) if parts else ""


def _count_fine_words(text: str, fine_words_lower: List[str]) -> int:
    """
    Count total occurrences of all fine words in text using pure string matching.
    Case-insensitive: caller should pass text already lowercased, and fine_words_lower already lowercased.

    Note:
      - Uses str.count(...) which counts NON-overlapping occurrences.
      - This intentionally does NOT tokenize/split the text.
    """
    if not text or not fine_words_lower:
        return 0
    total = 0
    for w in fine_words_lower:
        if w:
            total += text.count(w)
    return total


def _count_fine_words_token_mode(text_lower: str) -> int:
    """
    Token-based counting:
      - lowercased input
      - translate punct -> spaces
      - split on whitespace
      - exact token match against fine word set
    """
    if not text_lower or not _GLOBAL_FINE_WORDS_SET:
        return 0
    if _GLOBAL_PUNCT_TRANSLATION is None:
        tokens = text_lower.split()
    else:
        tokens = text_lower.translate(_GLOBAL_PUNCT_TRANSLATION).split()
    total = 0
    for tok in tokens:
        if tok in _GLOBAL_FINE_WORDS_SET:
            total += 1
    return total


def _process_single_line(line: str) -> Dict[str, Any]:
    """
    Helper function to process a single JSONL line (for multiprocessing).

    Returns:
        Dict with fields:
          - id
          - score (int)
          - (optional) counts: {fine_word: cnt}
    """
    try:
        item = json.loads(line.strip())
        text = _build_text_from_item(item, _GLOBAL_FIELDS)
        text_lower = text.lower()

        if _GLOBAL_MATCH_MODE == "token":
            score = _count_fine_words_token_mode(text_lower)
        else:
            score = _count_fine_words(text_lower, _GLOBAL_FINE_WORDS_LOWER)

        rec: Dict[str, Any] = {"id": item.get("id", ""), "score": score}

        if _GLOBAL_RETURN_COUNTS and _GLOBAL_FINE_WORDS_LOWER:
            # Keep per-word counts for debugging/analysis (optional; can be large).
            # Use original words (same length as fine_words_lower) if provided to keep nicer keys.
            keys = _GLOBAL_ORIG_FINE_WORDS if _GLOBAL_ORIG_FINE_WORDS else _GLOBAL_FINE_WORDS_LOWER
            counts = {}
            if _GLOBAL_MATCH_MODE == "token":
                if _GLOBAL_PUNCT_TRANSLATION is None:
                    tokens = text_lower.split()
                else:
                    tokens = text_lower.translate(_GLOBAL_PUNCT_TRANSLATION).split()
                # Build token frequency for fine words only (fast path).
                freq = {}
                for tok in tokens:
                    if tok in _GLOBAL_FINE_WORDS_SET:
                        freq[tok] = freq.get(tok, 0) + 1
                for k, w in zip(keys, _GLOBAL_FINE_WORDS_LOWER):
                    if w and w in freq:
                        counts[k] = freq[w]
            else:
                for k, w in zip(keys, _GLOBAL_FINE_WORDS_LOWER):
                    if w:
                        c = text_lower.count(w)
                        if c:
                            counts[k] = c
            rec["counts"] = counts

        return rec
    except Exception as e:
        return {
            "id": item.get("id", "unknown") if "item" in locals() else "unknown",
            "score": 0,
            "error": str(e),
        }


class LogicalWordCountScorer(BaseScorer):
    """
    LogicalWordCountScorer:
      1) Define a fine word set/list.
      2) For each sample, count occurrences of each fine word in (instruction + input + output),
         case-insensitive.
      3) Sum all counts as the sample score.
      4) Use pure string matching (NOT tokenization).
    """

    def _validate_config(self):
        # Fields validation (like TokenLengthScorer)
        if "fields" not in self.config or not isinstance(self.config["fields"], list) or len(self.config["fields"]) == 0:
            print("Warning: No fields specified in config. Using default fields: ['instruction', 'input', 'output'].")
            self.config["fields"] = ["instruction", "input", "output"]
        else:
            print(f"Using specified fields: {self.config['fields']}.")

        # Fine words config validation (loaded in _setup)
        if "fine_words" in self.config and not isinstance(self.config["fine_words"], (list, tuple)):
            print("Warning: fine_words should be a list/tuple of strings. Ignoring invalid fine_words.")
            self.config["fine_words"] = []

        if "fine_words_path" in self.config and self.config["fine_words_path"] is not None and not isinstance(
            self.config["fine_words_path"], str
        ):
            print("Warning: fine_words_path should be a string path. Ignoring invalid fine_words_path.")
            self.config["fine_words_path"] = None

        # Multiprocessing worker count validation
        if "max_workers" in self.config and isinstance(self.config["max_workers"], int) and self.config["max_workers"] > 0:
            print(f"Using specified max_workers: {self.config['max_workers']}.")
        else:
            default_workers = max(1, os.cpu_count() or 1)
            print(f"Warning: No/invalid max_workers, using default value of {default_workers} (CPU count).")
            self.config["max_workers"] = default_workers

        # Chunk size for streaming executor.map
        chunk_size = self.config.get("chunk_size", 2000)
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            chunk_size = 2000
        self.config["chunk_size"] = chunk_size

        # Optional: whether to emit per-word counts
        self.config["return_counts"] = bool(self.config.get("return_counts", False))

        # Match mode: substring (original requirement) or token (avoid substring matches)
        match_mode = self.config.get("match_mode", "substring")
        if match_mode not in ("substring", "token"):
            print("Warning: match_mode should be 'substring' or 'token'. Using default 'substring'.")
            match_mode = "substring"
        self.config["match_mode"] = match_mode

    def _load_fine_words(self) -> List[str]:
        words: List[str] = []

        cfg_words = self.config.get("fine_words", [])
        if isinstance(cfg_words, (list, tuple)):
            for w in cfg_words:
                if w is None:
                    continue
                s = str(w).strip()
                if s:
                    words.append(s)

        path = self.config.get("fine_words_path", None)
        if path:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        words.append(line)
            except Exception as e:
                print(f"Warning: failed to read fine_words_path={path}: {e}")

        # De-duplicate while preserving order
        seen = set()
        deduped: List[str] = []
        for w in words:
            if w not in seen:
                seen.add(w)
                deduped.append(w)
        return deduped

    def _setup(self):
        self._fine_words: List[str] = self._load_fine_words()
        self._fine_words_lower: List[str] = [w.lower() for w in self._fine_words]
        self._fields: List[str] = self.config.get("fields", ["instruction", "input", "output"])
        self._match_mode: str = self.config.get("match_mode", "substring")

        if not self._fine_words_lower:
            print("Warning: LogicalWordCountScorer has empty fine_words list; all scores will be 0.")
        else:
            print(f"Setting up LogicalWordCountScorer successfully (fine_words={len(self._fine_words_lower)})")

    def score_item(self, data_item: Dict) -> float:
        text = _build_text_from_item(data_item, self._fields)
        text_lower = text.lower()
        if self._match_mode == "token":
            # Keep single-process behavior consistent with worker tokenization.
            punct = (
                r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
                + "，。、；：？！“”‘’（）【】《》…—·"
            )
            trans = str.maketrans({ch: " " for ch in punct})
            tokens = text_lower.translate(trans).split()
            fine_set = set([w for w in self._fine_words_lower if w])
            return float(sum(1 for tok in tokens if tok in fine_set))
        return float(_count_fine_words(text_lower, self._fine_words_lower))

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        max_workers = self.config.get("max_workers", 1)
        return_counts = bool(self.config.get("return_counts", False))
        fields = self.config.get("fields", ["instruction", "input", "output"])
        match_mode = self.config.get("match_mode", "substring")

        print(f"Using {max_workers} worker(s) for parallel processing")

        with open(dataset, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line for line in f]

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(self._fine_words_lower, return_counts, self._fine_words, fields, match_mode),
        ) as executor:
            results = list(
                tqdm(
                    executor.map(_process_single_line, lines),
                    total=num_lines,
                    desc=self.config.get("name", "LogicalWordCountScorer"),
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
        chunk_size = self.config.get("chunk_size", 2000)
        return_counts = bool(self.config.get("return_counts", False))
        fields = self.config.get("fields", ["instruction", "input", "output"])
        match_mode = self.config.get("match_mode", "substring")

        done_ids = set()
        if resume and os.path.exists(output_file):
            repair_trailing_incomplete_jsonl(output_file)
            done_ids = load_jsonl_id_set(output_file, id_key="id")
            if done_ids:
                print(f"[LogicalWordCountScorer] Resume enabled. Found {len(done_ids)} unique completed ids in {output_file}.")

        if not resume:
            out_dir = os.path.dirname(output_file)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(output_file, "w", encoding="utf-8"):
                pass

        if not isinstance(chunk_size, int) or chunk_size <= 0:
            chunk_size = 2000

        print(f"Using {max_workers} worker(s) for parallel processing")
        pbar = tqdm(total=num_lines, desc=self.config.get("name", "LogicalWordCountScorer"))

        buf_records: List[Dict[str, Any]] = []
        chunk_lines: List[str] = []

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(self._fine_words_lower, return_counts, self._fine_words, fields, match_mode),
        ) as executor:
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

                    chunk_lines.append(line)

                    if len(chunk_lines) >= chunk_size:
                        for rec in executor.map(_process_single_line, chunk_lines):
                            buf_records.append(rec)
                            rid = normalize_id(rec.get("id", ""))
                            if rid:
                                done_ids.add(rid)

                            if len(buf_records) >= 1000:
                                append_jsonl(buf_records, output_file, flush=True)
                                buf_records.clear()

                            pbar.update(1)
                        chunk_lines.clear()

            if chunk_lines:
                for rec in executor.map(_process_single_line, chunk_lines):
                    buf_records.append(rec)
                    rid = normalize_id(rec.get("id", ""))
                    if rid:
                        done_ids.add(rid)

                    if len(buf_records) >= 1000:
                        append_jsonl(buf_records, output_file, flush=True)
                        buf_records.clear()

                    pbar.update(1)
                chunk_lines.clear()

        if buf_records:
            append_jsonl(buf_records, output_file, flush=True)

        pbar.close()
        return output_file
