#!/usr/bin/env python3
"""Robust PDF Title/Outline Extractor
   - headers/footers aware
   - form-aware
   - multi-line titles (with guards for recipe PDFs & RFP/cover pages)
   - resilient fallbacks for uniform-layout docs (e.g., recipe PDFs)
"""

import argparse
import json
import re
import sys
import warnings
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import pdfplumber
from jsonschema import validate
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# -------------------------
# Config / constants
# -------------------------
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEBUG = True
SCHEMA_PATH = Path("sample_dataset/schema/output_schema.json")
MAX_HEADING_LENGTH = 150

# Heuristic thresholds
FONT_REL_PRIMARY = 1.20
FONT_REL_BOLD = 1.05
FONT_REL_CENTERED = 1.30
GLOBAL_SIZE_BUMP = 1.20   # for bold absolute-size rescue
ABS_SIZE_MIN = 14.0        # bold absolute-size rescue

# How similar two sizes must be to be considered the "same" level
SIZE_LEVEL_TOL = 1.2


def log_debug(msg: str):
    if DEBUG:
        print(f"[DEBUG] {msg}", file=sys.stderr)


def log_warn(msg: str):
    print(f"[WARN] {msg}", file=sys.stderr)


# -------------------------
# Text utilities
# -------------------------

def normalize_text(t: str) -> str:
    if not t:
        return t
    # collapse 3+ repeated characters (helps with OCR like "Reeeequest")
    t = re.sub(r'(.)\1{2,}', r'\1', t)
    # collapse whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def upper_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0


def is_title_case(text: str) -> bool:
    words = re.findall(r"[A-Za-z]+", text)
    return bool(words) and all(w[0].isupper() for w in words)


def is_sentence_like(text: str) -> bool:
    if not text:
        return False
    verbs = r'\b(is|are|was|were|be|been|being|provides?|includes?|contains?|must|should|shall|will|can|aims|designed|intended)\b'
    if re.search(verbs, text, re.I) and len(text.split()) >= 8:
        return True
    if text.strip().endswith('.') and len(text.split()) >= 6:
        return True
    return False


def is_valid_heading_text(text: str) -> bool:
    if not text or not text.strip():
        return False
    txt = text.strip()
    if len(txt) > MAX_HEADING_LENGTH:
        return False
    if txt[0] in {'•', '·', '▪', '–', '-'}:
        return False
    if re.fullmatch(r'[\d\W]+', txt):
        return False
    if re.search(r'[.]{3,}|[-]{3,}|\s+\d+\s*$', txt):
        return False
    # very short ALL-CAPS usually noise
    if txt.isupper() and len(txt) < 15:
        return False
    if is_sentence_like(txt):
        return False
    if txt.rstrip().endswith('.'):
        return False
    return True


# -------------------------
# Ingredient-like detector (for recipe PDFs)
# -------------------------
ING_UNITS = {
    "cup", "cups", "tbsp", "tablespoon", "tablespoons", "tsp", "teaspoon", "teaspoons",
    "g", "kg", "ml", "l", "oz", "pound", "pounds", "lb", "lbs",
}
ING_WORDS = {
    "salt", "pepper", "garlic", "onion", "butter", "oil", "water", "flour", "sugar", "egg", "eggs",
    "tomato", "tomatoes", "lemon", "milk", "yogurt", "paprika", "cumin", "turmeric", "cilantro",
    "broth", "vinegar", "mustard", "cheese", "cream", "mayonnaise", "coconut", "rice", "chili", "lime",
}
SPECIAL_ING_PHRASES = {"oil for frying", "to taste", "for serving", "pinch of"}


def is_ingredient_like(s: str) -> bool:
    if not s:
        return False
    t = s.strip().lower()
    if not t:
        return False

    if "ingredients" in t or "instructions" in t:
        return True
    if any(ph in t for ph in SPECIAL_ING_PHRASES):
        return True

    has_number = bool(re.search(r"\d", t))
    has_unit = any(re.search(rf"\b{re.escape(u)}\b", t) for u in ING_UNITS)
    has_ing = any(re.search(rf"\b{re.escape(w)}\b", t) for w in ING_WORDS)

    if has_number and (has_unit or has_ing):
        return True

    tokens = re.findall(r"[A-Za-z][A-Za-z’'\-]*", t)
    if 1 <= len(tokens) <= 6 and has_ing:
        return True

    return False


# -------------------------
# Line segmentation (split wide rows into segments)
# -------------------------

def split_line_segments(line_words: List[Dict[str, Any]], gap_multiplier: float = 1.8):
    if not line_words:
        return []
    words_sorted = sorted(line_words, key=lambda d: d["x0"])
    sizes = [w.get("size", w.get("height", 0)) for w in words_sorted]
    size = statistics.median(sizes) if sizes else 10
    gap_threshold = size * gap_multiplier

    segments = []
    current = [words_sorted[0]]
    prev_x1 = words_sorted[0]["x1"]

    for w in words_sorted[1:]:
        gap = w["x0"] - prev_x1
        if gap > gap_threshold:
            segments.append(current)
            current = [w]
        else:
            current.append(w)
        prev_x1 = w["x1"]
    segments.append(current)
    return segments


# -------------------------
# Block extraction
# -------------------------

def extract_blocks(pdf_path: Path) -> List[Dict[str, Any]]:
    blocks = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            log_debug(f"Processing {pdf_path.name} ({len(pdf.pages)} pages)")
            for pno, page in enumerate(pdf.pages):
                words = page.extract_words(
                    x_tolerance=1,
                    y_tolerance=1,
                    keep_blank_chars=False,
                    extra_attrs=["size", "fontname"],
                )
                if not words:
                    continue

                line_map: Dict[float, List[Dict[str, Any]]] = {}
                for w in words:
                    line_map.setdefault(round(w["top"], 1), []).append(w)

                for top, line_words in sorted(line_map.items(), key=lambda x: x[0]):
                    segments = split_line_segments(line_words, 1.8)
                    seg = sorted(segments, key=lambda s: min(w["x0"] for w in s))[0]

                    text = " ".join(w["text"] for w in sorted(seg, key=lambda d: d["x0"]))
                    text = normalize_text(text)
                    if not is_valid_heading_text(text):
                        continue

                    sizes = [w.get("size", w.get("height", 0)) for w in seg]
                    if not sizes:
                        continue
                    size = statistics.median(sizes)
                    fontname = seg[0].get("fontname", "")
                    bold = bool(re.search(r"(Bold|Semibold|Black)", fontname, re.I))

                    x0 = float(min(w["x0"] for w in seg))
                    x1 = float(max(w["x1"] for w in seg))
                    page_width = page.width or 595
                    centred = abs((page_width / 2) - ((x0 + x1) / 2)) < page_width * 0.15

                    blocks.append(
                        {
                            "text": text,
                            "size": float(size),
                            "bold": bool(bold),
                            "centred": bool(centred),
                            "x0": x0,
                            "x1": x1,
                            "y0": float(top),
                            "page": int(pno),
                        }
                    )
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}", file=sys.stderr)

    log_debug(f"Extracted {len(blocks)} candidate blocks")
    return blocks


# -------------------------
# Repeating header/footer cleaner (position-aware)
# -------------------------

def remove_repeating_headers_footers(blocks: List[Dict[str, Any]], min_pages: int = 3) -> List[Dict[str, Any]]:
    from collections import defaultdict
    ybins = defaultdict(lambda: defaultdict(set))
    for b in blocks:
        ybin = round(b["y0"] / 10) * 10
        ybins[b["text"]][ybin].add(b["page"])

    header_keys = {(t, ybin) for t, ymap in ybins.items() for ybin, pages in ymap.items() if len(pages) >= min_pages}
    out = [b for b in blocks if (b["text"], round(b["y0"] / 10) * 10) not in header_keys]
    if len(out) != len(blocks):
        log_debug(f"Removed {len(blocks) - len(out)} header/footer occurrences")
    return out


# -------------------------
# Title extraction helpers (RFP/cover-aware)
# -------------------------
COVER_TITLE_HINTS = [
    r"\bRFP\b",
    r"\bRequest\s+for\s+Proposal\b",
    r"\bRequest\s+for\s+Information\b",
    r"\bRequest\s+for\s+Quotation\b",
    r"\bTerms\s+of\s+Reference\b",
]


def _blocks_page0(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted([b for b in blocks if b.get("page", 0) == 0], key=lambda x: (x.get("y0", 0.0), x.get("x0", 0.0)))


def _is_cover_title_seed(text: str) -> bool:
    t = normalize_text(text or "")
    return any(re.search(pat, t, re.I) for pat in COVER_TITLE_HINTS)


def _compose_nearby_centered_lines(page0: List[Dict[str, Any]], start_idx: int) -> Tuple[str, float]:
    """Join the seed line with immediate centered (or similarly left-aligned) lines below."""
    seed = page0[start_idx]
    parts = [seed["text"]]
    y_top = seed["y0"]
    max_size = seed.get("size", 0.0)

    i = start_idx + 1
    while i < len(page0):
        b = page0[i]
        if b["y0"] - page0[i - 1]["y0"] > max(18.0, max_size * 1.6):
            break
        # prefer centered OR same left alignment; avoid sentences/dates/obvious junk
        if not (b.get("centred", False) or abs(b.get("x0", 0.0) - seed.get("x0", 0.0)) <= 14.0):
            break
        if abs(b.get("size", 0.0) - max_size) > 2.2:
            break
        txt = b["text"].strip()
        if not txt or is_sentence_like(txt):
            break
        if re.search(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b\s+\d{4}", txt):
            break
        # keep building until we reach a reasonable title length
        if len(" ".join(parts + [txt]).split()) > 28:
            break
        parts.append(txt)
        i += 1

    title = normalize_text("  ".join(parts))
    return (title, y_top)


def extract_primary_title(blocks: List[Dict[str, Any]]) -> str:
    first = [b for b in blocks if b.get("page", 0) == 0]
    if not first:
        return ""
    cands = [b for b in first if not is_ingredient_like(b["text"])] or first
    title_block = max(cands, key=lambda x: x.get("size", 0.0))
    return title_block["text"].strip()


def extract_composite_title(blocks: List[Dict[str, Any]]) -> str:
    first = [b for b in blocks if b.get("page", 0) == 0]
    if not first:
        return ""
    cands = [b for b in first if not is_ingredient_like(b["text"])] or first
    max_size = max(b.get("size", 0.0) for b in cands)
    base = sorted(
        [b for b in cands if (max_size - b.get("size", 0.0) <= 0.75)
         and (b.get("centred", False) or b.get("y0", 0.0) < min(bb["y0"] for bb in cands) + 200)
         and not is_ingredient_like(b["text"])],
        key=lambda b: (b.get("y0", 0.0), b.get("x0", 0.0)),
    )
    if not base:
        base = [max(cands, key=lambda x: x.get("size", 0.0))]

    y_last = max(b.get("y0", 0.0) for b in base)
    extensions = []
    for b in sorted(cands, key=lambda x: (x.get("y0", 0.0), x.get("x0", 0.0))):
        if b.get("y0", 0.0) <= y_last:
            continue
        if b.get("y0", 0.0) > y_last + 120:
            break
        if (max_size - b.get("size", 0.0) > 8) or not b.get("centred", False):
            continue
        if is_ingredient_like(b["text"]) or re.search(r"^ingredients?:?\b|^instructions?:?\b", b["text"], re.I):
            break
        extensions.append(b)
        y_last = b.get("y0", 0.0)
        if len(extensions) >= 2:
            break

    parts: List[str] = []
    seen = set()
    for b in sorted(base + extensions, key=lambda x: (x.get("y0", 0.0), x.get("x0", 0.0))):
        if b["text"] not in seen:
            parts.append(b["text"])
            seen.add(b["text"])

    title = "  ".join(parts).strip()
    if upper_ratio(title) > 0.8 and len(title.split()) <= 6:
        return ""
    if is_ingredient_like(title):
        return ""
    if len(title.split()) > 28:
        return ""
    return title


def _cover_keyword_block_index(page0: List[Dict[str, Any]]) -> Optional[int]:
    for i, b in enumerate(page0):
        t = normalize_text(b["text"]).lower()
        if "request for proposal" in t or re.search(r"\brfp\b", t):
            return i
    return None


def compute_title_info(blocks: List[Dict[str, Any]], single: bool) -> Tuple[str, Optional[float]]:
    """Return (title_text, title_y0_on_page0 or None)."""
    if single:
        first_page_blocks = [b for b in blocks if b.get("page", 0) == 0]
        non_ing = [b for b in first_page_blocks if not is_ingredient_like(b["text"]) and b.get("centred", False)]
        if not non_ing:
            non_ing = [b for b in first_page_blocks if not is_ingredient_like(b["text"])]
        base = non_ing if non_ing else first_page_blocks
        if not base:
            return ("", None)
        top = max(base, key=lambda x: x.get("size", 0.0))
        posterish = (upper_ratio(top["text"]) > 0.7 and len(top["text"].split()) <= 8) or any(
            k in top["text"].upper() for k in ["INVITATION", "PARTY", "CONCERT", "SALE"]
        )
        return ("" if posterish else top["text"].strip(), float(top.get("y0", 0.0)))

    # Multi-page: try RFP/cover-aware assembly first
    page0 = _blocks_page0(blocks)

    # Strategy A: explicit seed hint lines
    for idx, b in enumerate(page0):
        if _is_cover_title_seed(b["text"]):
            title, y0 = _compose_nearby_centered_lines(page0, idx)
            if title and not is_ingredient_like(title):
                return (title, y0)

    # Strategy B: keyword search then stitch downward even if not centered
    kidx = _cover_keyword_block_index(page0)
    if kidx is not None:
        parts = [page0[kidx]["text"]]
        y0 = page0[kidx]["y0"]
        max_size = page0[kidx].get("size", 0.0)
        j = kidx + 1
        while j < len(page0):
            b = page0[j]
            if b["y0"] - page0[j - 1]["y0"] > max(22.0, max_size * 1.8):
                break
            if abs(b.get("size", 0.0) - max_size) > 2.5:
                break
            t = b["text"].strip()
            if not t or is_sentence_like(t):
                break
            if len(" ".join(parts + [t]).split()) > 30:
                break
            parts.append(t)
            j += 1
        title = normalize_text("  ".join(parts))
        if title and not is_ingredient_like(title):
            return (title, y0)

    # Fallback to composite/primary
    title = extract_composite_title(blocks) or extract_primary_title(blocks)
    y0 = None
    if page0 and title:
        for b in page0:
            if b["text"].strip() in {p.strip() for p in title.split("  ")}:  # locate one of the parts
                y0 = float(b.get("y0", 0.0))
                break
    return (title, y0)


# -------------------------
# Form detection
# -------------------------

def is_form_document(blocks: List[Dict[str, Any]]) -> bool:
    if not blocks:
        return False
    texts = [b["text"] for b in blocks if b.get("page", 0) == 0]
    count_numbered = sum(1 for t in texts if re.match(r'^\d+(\.| )', t))
    form_keywords = [
        "S.No", "Signature of", "Date", "Designation", "Home Town",
        "Amount of advance", "PAY +", "Service Book"
    ]
    kw_hit = any(any(k.lower() in t.lower() for k in form_keywords) for t in texts)
    sizes = [b.get("size", 0.0) for b in blocks]
    std = statistics.pstdev(sizes) if len(sizes) > 1 else 0.0
    return (count_numbered >= 5 and kw_hit) or (kw_hit and std < 1.5 and len({b.get("page", 0) for b in blocks}) == 1)


# -------------------------
# Level assignment (global) with size tolerance
# -------------------------

def assign_levels_global(headings_df: pd.DataFrame, max_levels: int = 6) -> pd.DataFrame:
    """Assign levels by grouping near-equal sizes, with an optional cap on the
    number of distinct levels (useful for RFP docs where we expect ~H1–H4)."""
    df = headings_df.copy()
    # group near-equal sizes into the same logical bin
    sizes_sorted = sorted(df["size"].unique(), reverse=True)
    bins: List[float] = []
    for s in sizes_sorted:
        if not bins or all(abs(s - b) > SIZE_LEVEL_TOL for b in bins):
            bins.append(float(s))

    # initial nearest-bin assignment
    def nearest_bin(val: float) -> float:
        return min(bins, key=lambda b: abs(val - b)) if bins else val
    df["_size_bin"] = df["size"].apply(nearest_bin)

    # limit the number of distinct bins if needed by merging the smallest bins upward
    uniq = sorted(df["_size_bin"].unique(), reverse=True)
    while len(uniq) > max_levels:
        # take the smallest bin and merge it into the next-smallest
        smallest = min(uniq)
        larger_bins = sorted([u for u in uniq if u > smallest])
        if not larger_bins:
            break
        target = larger_bins[-1]  # nearest larger (the smallest among the larger set)
        df.loc[df["_size_bin"] == smallest, "_size_bin"] = target
        uniq = sorted(df["_size_bin"].unique(), reverse=True)

    uniq = sorted(df["_size_bin"].unique(), reverse=True)
    level_map = {s: f"H{min(i + 1, 6)}" for i, s in enumerate(uniq)}
    df["level"] = df["_size_bin"].map(level_map)
    return df.drop(columns=["_size_bin"])


# -------------------------
# Optional OCR fallback
# -------------------------
try:
    import pytesseract
    from PIL import Image  # noqa: F401
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


def extract_ocr_blocks(pdf_path: Path, max_pages: int = 2, dpi: int = 200) -> List[Dict[str, Any]]:
    if not OCR_AVAILABLE:
        return []
    results: List[Dict[str, Any]] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for pno, page in enumerate(pdf.pages[:max_pages]):
                try:
                    im = page.to_image(resolution=dpi).original
                except Exception as e:
                    log_debug(f"OCR rasterization failed on page {pno}: {e}")
                    continue
                import pytesseract
                data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT)
                n = len(data["text"])
                lines_map: Dict[tuple, List[Dict[str, Any]]] = {}
                for i in range(n):
                    txt = data["text"][i]
                    if not txt or not txt.strip():
                        continue
                    try:
                        conf = int(data["conf"][i])
                    except Exception:
                        conf = 0
                    if conf < 40:
                        continue
                    key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                    lines_map.setdefault(key, []).append(
                        {
                            "text": txt,
                            "x0": data["left"][i],
                            "y0": data["top"][i],
                            "x1": data["left"][i] + data["width"][i],
                            "y1": data["top"][i] + data["height"][i],
                            "h": data["height"][i],
                        }
                    )

                for _, words in lines_map.items():
                    words_sorted = sorted(words, key=lambda w: w["x0"])
                    text = normalize_text(" ".join(w["text"] for w in words_sorted))
                    if not is_valid_heading_text(text):
                        continue
                    x0 = float(min(w["x0"] for w in words_sorted))
                    x1 = float(max(w["x1"] for w in words_sorted))
                    y0 = float(min(w["y0"] for w in words_sorted))
                    h = float(statistics.median([w["h"] for w in words_sorted]))
                    centred = abs((im.width / 2) - ((x0 + x1) / 2)) < im.width * 0.15
                    results.append(
                        {
                            "text": text,
                            "size": h,
                            "bold": False,
                            "centred": centred,
                            "x0": x0,
                            "x1": x1,
                            "y0": y0,
                            "page": int(pno),
                        }
                    )
    except Exception as e:
        log_debug(f"OCR pass failed: {e}")
    if results:
        log_debug(f"OCR extracted {len(results)} line candidates")
    return results


# -------------------------
# Feature extraction
# -------------------------

def extract_features(blocks: List[Dict[str, Any]]) -> pd.DataFrame:
    if not blocks:
        return pd.DataFrame()
    df = pd.DataFrame(blocks).sort_values(["page", "y0"]).reset_index(drop=True)
    body_medians = df.groupby("page")["size"].median().rename("body")
    df = df.join(body_medians, on="page")
    df["font_rel"] = df["size"] / df["body"].clip(lower=1e-3)
    df["n_chars"] = df["text"].str.len()
    df["n_words"] = df["text"].str.split().str.len()
    df["upper_ratio"] = df["text"].apply(upper_ratio)
    df["title_case"] = df["text"].apply(lambda x: int(is_title_case(x)))
    df["gap"] = df.groupby("page")["y0"].diff().fillna(0.0)
    df["gap_norm"] = df["gap"] / df["size"].clip(lower=1e-3)
    return df


FEATURES = ["font_rel", "bold", "centred", "n_words", "upper_ratio", "title_case", "gap_norm"]


# -------------------------
# Helpers
# -------------------------

def _ensure_col(df_like: pd.DataFrame, src: Optional[pd.DataFrame], col: str, default=0):
    if col not in df_like.columns:
        if src is not None and isinstance(src, pd.DataFrame) and col in src.columns:
            try:
                df_like[col] = src.loc[df_like.index, col]
            except Exception:
                df_like[col] = default
        else:
            df_like[col] = default


def _get_text_series(df_like: pd.DataFrame, fallback_df: pd.DataFrame) -> pd.Series:
    if "text" in df_like.columns:
        return df_like["text"].astype(str)
    return fallback_df.loc[df_like.index, "text"].astype(str)


def _ensure_nav_cols(df_like: pd.DataFrame, src: Optional[pd.DataFrame]):
    for col, default in [("page", 0), ("y0", 0.0), ("x0", 0.0), ("size", 0.0), ("centred", False)]:
        _ensure_col(df_like, src, col, default=default)


# -------------------------
# Fallback: infer headings around "Ingredients"
# -------------------------
ING_RE = re.compile(r'^\s*ingredients?\s*:?\s*$', re.I)
INS_RE = re.compile(r'^\s*instructions?\s*:?\s*$', re.I)


def recipe_like_fallback(pdf_path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for pno, page in enumerate(pdf.pages):
                raw = page.extract_text() or ""
                lines = [ln.rstrip() for ln in raw.splitlines()]
                if not lines:
                    continue
                for idx, ln in enumerate(lines):
                    ln_clean = ln.lstrip("•- \t").strip()
                    if ING_RE.match(ln_clean):
                        j = idx - 1
                        while j >= 0 and idx - j <= 6:
                            cand_raw = lines[j]
                            cand = cand_raw.lstrip("•- \t").strip()
                            if not cand:
                                j -= 1
                                continue
                            if INS_RE.match(cand) or ING_RE.match(cand):
                                j -= 1
                                continue
                            if is_ingredient_like(cand):
                                j -= 1
                                continue
                            wc = len(re.findall(r"[A-Za-z][A-Za-z’'\-]*", cand))
                            if 2 <= wc <= 10 and not cand.endswith(('.', ':', ';')):
                                items.append({"level": "H1", "text": cand, "page": pno})
                            break
    except Exception as e:
        log_debug(f"recipe_like_fallback failed: {e}")

    out = []
    seen = set()
    for it in items:
        key = (it["text"].lower(), it["page"])
        if key not in seen:
            out.append(it)
            seen.add(key)
    return out


# -------------------------
# Outline utilities
# -------------------------

def dedup_outline_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for it in items or []:
        txt = (it.get("text") or "").strip()
        if not txt:
            continue
        pg = int(it.get("page", 0))
        key = (txt.lower(), pg)
        if key in seen:
            continue
        out.append({
            "level": it.get("level", "H1"),
            "text": txt,
            "page": pg,
        })
        seen.add(key)
    return out


def merge_wrapped_headings(cand: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive lines that are likely the same heading (wrap across lines).
       NOTE: allow colon-terminated first lines (e.g., "Appendix B:" + rest).
    """
    if cand.empty:
        return cand
    cand = cand.sort_values(["page", "y0"]).copy()
    idx = list(cand.index)
    to_drop = []
    for i in range(len(idx) - 1):
        a = cand.loc[idx[i]]
        b = cand.loc[idx[i + 1]]
        try:
            same_page = int(a["page"]) == int(b["page"])
            if not same_page:
                continue
            ygap = float(b["y0"]) - float(a["y0"])
            if ygap <= 0:
                continue
            size_close = abs(float(a["size"]) - float(b["size"])) <= 1.1
            align_close = (bool(a["centred"]) and bool(b["centred"])) or abs(float(a["x0"]) - float(b["x0"])) <= 16.0
            if not (size_close and align_close and ygap <= max(float(a["size"]), float(b["size"])) * 1.8):
                continue
            a_txt = str(a.get("text", "")).strip()
            b_txt = str(b.get("text", "")).strip()
            if not a_txt or not b_txt:
                continue
            # allow colon at the end, but stop for period/semicolon
            if re.search(r'[.;]\s*$', a_txt):
                continue
            merged = normalize_text(f"{a_txt} {b_txt}")
            if len(merged.split()) > 26 or is_sentence_like(merged):
                continue
            # perform merge: keep A, drop B
            cand.at[idx[i], "text"] = merged
            cand.at[idx[i], "size"] = max(float(a["size"]), float(b["size"]))
            to_drop.append(idx[i + 1])
        except Exception:
            continue
    if to_drop:
        cand = cand.drop(index=list(set(to_drop)))
    return cand


def include_colon_subheads(df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Loosened rule to include smaller colon-terminated subheads (e.g., H3/H4 like
       "For each Ontario citizen it could mean:") even if font_rel smaller, as long as
       they're bold OR modestly above body size.
    """
    if df.empty:
        return df
    _ensure_nav_cols(df, base)
    txt = _get_text_series(base, base)

    # candidates from the full feature frame (base)
    colon_like = base.copy()
    tser = _get_text_series(colon_like, base)
    colon_like = colon_like[
        tser.str.rstrip().str.endswith(":")
        & (~tser.apply(is_sentence_like))
        & (~tser.apply(is_ingredient_like))
        & (colon_like["n_words"] >= 3)
        & (colon_like["n_words"] <= 16)
        & (
            (colon_like["bold"]) |
            (colon_like["font_rel"] >= 1.05)
        )
    ]
    if colon_like.empty:
        return df

    # add missing colon subheads that aren't already in df (by text,page)
    have = set((str(t).strip().lower(), int(p)) for t, p in zip(df.get("text", []), df.get("page", [])))
    add_rows = []
    for _, r in colon_like.iterrows():
        key = (str(r["text"]).strip().lower(), int(r["page"]))
        if key in have:
            continue
        add_rows.append(r)
    if add_rows:
        df = pd.concat([df, pd.DataFrame(add_rows)], ignore_index=True)
    return df


def harmonize_top_pair_page1(df: pd.DataFrame) -> pd.DataFrame:
    """On page 1, if the first two headings are close in size/position, coerce them
    into the same level by snapping their 'size' up to the larger of the two."""
    if df.empty:
        return df
    d = df.sort_values(["page", "y0"]).copy()
    p1 = d[d["page"].astype(int) == 1]
    if len(p1) < 2:
        return df
    first_two = p1.head(2)
    s1, s2 = float(first_two.iloc[0]["size"]), float(first_two.iloc[1]["size"])
    y1, y2 = float(first_two.iloc[0]["y0"]), float(first_two.iloc[1]["y0"])
    x1, x2 = float(first_two.iloc[0].get("x0", 0.0)), float(first_two.iloc[1].get("x0", 0.0))
    c1, c2 = bool(first_two.iloc[0].get("centred", False)), bool(first_two.iloc[1].get("centred", False))

    y_close = (y1 < 350 and y2 < 380) and (y2 - y1) <= 200
    size_close = abs(s1 - s2) <= 2.0
    align_ok = (c1 and c2) or abs(x1 - x2) <= 40.0
    if y_close and size_close and align_ok:
        bigger = max(s1, s2)
        idxs = first_two.index.tolist()
        if s1 < bigger:
            df.at[idxs[0], "size"] = bigger
        if s2 < bigger:
            df.at[idxs[1], "size"] = bigger
    return df
# -------------------------
# Title extraction helpers (RFP/cover-aware)
# -------------------------
COVER_TITLE_HINTS = [
    r"\bRFP\b",
    r"\bRequest\s+for\s+Proposal\b",
    r"\bRequest\s+for\s+Information\b",
    r"\bRequest\s+for\s+Quotation\b",
    r"\bTerms\s+of\s+Reference\b",
]


def detect_rfp(blocks: List[Dict[str, Any]], title: str) -> bool:
    """Heuristic to detect RFP/cover-style documents."""
    t = (title or "")
    if re.search(r"\bRFP\b|\bRequest\s+for\s+Proposal\b", t, re.I):
        return True
    page0 = [b for b in blocks if int(b.get("page", 0)) == 0]
    for b in page0:
        if _is_cover_title_seed(b.get("text", "")):
            return True
    return False


# -------------------------
# Core processing
# -------------------------

def is_single_page(pdf_path: Path) -> bool:
    with pdfplumber.open(str(pdf_path)) as pdf:
        return len(pdf.pages) == 1


def process_file(pdf_path: Path, model_data=None, one_based: bool = False) -> Dict[str, Any]:
    try:
        # 1) Extract text blocks & clean headers/footers
        blocks = extract_blocks(pdf_path)
        blocks = remove_repeating_headers_footers(blocks)

        if not blocks:
            outline = recipe_like_fallback(pdf_path)
            title = ""
            if one_based:
                for it in outline:
                    it["page"] += 1
            return {"title": title, "outline": outline}

        # 2) Title (cover-aware for multi-page)
        single = is_single_page(pdf_path)
        title, title_y0 = compute_title_info(blocks, single)
        is_rfp = detect_rfp(blocks, title)

        # 3) Forms → only title
        if is_form_document(blocks):
            return {"title": extract_primary_title(blocks), "outline": []}

        # 4) Features
        df = extract_features(blocks)
        if df.empty:
            outline = recipe_like_fallback(pdf_path)
            if one_based:
                for it in outline:
                    it["page"] += 1
            if is_ingredient_like(title) or len(title.split()) > 25:
                title = ""
            return {"title": title, "outline": outline}

        global_median_size = df["size"].median()

        # 5) Predict via model (if provided)
        headings_df = pd.DataFrame()
        if model_data is not None:
            try:
                if isinstance(model_data, dict):
                    model = model_data.get("model")
                    feature_order = model_data.get("feature_order", FEATURES)
                else:
                    model = model_data
                    feature_order = FEATURES

                X = np.zeros((len(df), len(feature_order)))
                for i, feat in enumerate(feature_order):
                    X[:, i] = df.get(feat, pd.Series(0, index=df.index)).values

                proba = model.predict_proba(X)[:, 1]
                headings_df = df[proba > 0.3].copy()
                log_debug(f"Model proposed {len(headings_df)} headings")
            except Exception as e:
                print(f"Model prediction error: {e}", file=sys.stderr)

        # 6) Heuristic fallback / refinement
        if headings_df.empty:
            cand = df[
                (df["font_rel"] > FONT_REL_PRIMARY)
                | (
                    (df["bold"])
                    & (
                        (df["font_rel"] > FONT_REL_BOLD)
                        | (df["size"] >= global_median_size * GLOBAL_SIZE_BUMP)
                        | (df["size"] >= ABS_SIZE_MIN)
                    )
                )
                | ((df["centred"]) & (df["font_rel"] > FONT_REL_CENTERED))
            ].copy()
        else:
            cand = headings_df.copy()

        # ensure navigation columns are present
        _ensure_nav_cols(cand, df)

        # --- robust text filtering ---
        txt = _get_text_series(cand, df)
        cand = cand[~txt.apply(is_sentence_like)]
        txt = _get_text_series(cand, df)
        cand = cand[~txt.str.strip().str.endswith(".")]
        # drop ingredient-like fragments
        txt = _get_text_series(cand, df)
        cand = cand[~txt.apply(is_ingredient_like)]
        _ensure_nav_cols(cand, df)

        # drop page-0 junk ABOVE the detected title line (cover slogans)
        if not single and title and title_y0 is not None and not cand.empty:
            cand = cand[~((cand["page"].astype(int) == 0) & (cand["y0"] < float(title_y0) - 2.0))]

        # drop page-0 title parts (defensive)
        if title and not cand.empty:
            try:
                title_parts = [p.strip() for p in title.split("  ") if p.strip()]
                if title_parts:
                    txt = _get_text_series(cand, df)
                    mask = (cand["page"].astype(int) == 0) & (txt.isin(title_parts))
                    cand = cand.loc[~mask]
            except Exception as e:
                log_debug(f"title-part filter skipped: {e}")

        _ensure_nav_cols(cand, df)

        # single-page posters: keep strongest top line
        if single and not cand.empty:
            p0 = cand[cand["page"].astype(int) == 0].copy()
            if not p0.empty:
                page_max = p0["size"].max()
                near = p0[p0["size"] >= page_max - 0.6]
                keep_idx = near.sort_values("y0").head(1).index
                cand = cand.drop(index=p0.index.difference(keep_idx))

        # allow more subheads that end with colon
        cand = include_colon_subheads(cand, df)

        # merge wrapped headings before dedupe/levels
        if not cand.empty:
            cand = merge_wrapped_headings(cand)

        # A small normalization for page 1: two top headings should align as same level when sizes are close
        cand = harmonize_top_pair_page1(cand)

        # Build outline
        outline: List[Dict[str, Any]] = []
        if not cand.empty:
            txt = _get_text_series(cand, df)
            cand = cand.assign(text=txt.values)
            cand = cand.drop_duplicates(subset=["text", "page"])  # dedupe exact (text,page)
            cand = assign_levels_global(cand, max_levels=(4 if is_rfp else 6))
            outline = cand[["level", "text", "page"]].to_dict("records")

        # 7) Union with recipe-like fallback on multi-page docs
        if not single:
            recipe_outline = recipe_like_fallback(pdf_path)
            if recipe_outline:
                outline = dedup_outline_items((outline or []) + recipe_outline)

        # 8) OCR fallback for single-page
        if not outline and single:
            ocr_blocks = extract_ocr_blocks(pdf_path)
            if ocr_blocks:
                df_ocr = extract_features(ocr_blocks)
                if not df_ocr.empty:
                    global_med_ocr = df_ocr["size"].median()
                    sel = df_ocr[
                        (df_ocr["font_rel"] > FONT_REL_PRIMARY)
                        | ((df_ocr["centred"]) & (df_ocr["font_rel"] > FONT_REL_CENTERED))
                        | ((df_ocr["size"] >= global_med_ocr * GLOBAL_SIZE_BUMP) & (df_ocr["n_words"] <= 8))
                    ]
                    txt = _get_text_series(sel, df_ocr)
                    sel = sel.assign(text=txt.values)
                    sel = sel[~txt.apply(is_sentence_like)]
                    if not sel.empty:
                        sel = assign_levels_global(sel)
                        outline = sel[["level", "text", "page"]].to_dict("records")

        # 9) Keep legitimate cover titles; only blank clearly suspicious ones (ingredient-like or extremely long)
        if title:
            if (not is_rfp) and (is_ingredient_like(title) or len(title.split()) > 40):
                title = ""

        # 10) For cover-based docs, prefer outline starting from page 1 if there is content there
        if not single and outline:
            has_p1 = any(it["page"] == 1 for it in outline)
            if has_p1:
                outline = [it for it in outline if it["page"] != 0]

        # 11) page numbering option
        if one_based:
            for item in outline:
                item["page"] += 1

        return {"title": title, "outline": outline}

    except Exception as e:
        log_warn(f"process_file failed on {pdf_path.name}: {e}")
        try:
            outline = recipe_like_fallback(pdf_path)
            return {"title": "", "outline": outline}
        except Exception:
            return {"title": "", "outline": []}


# -------------------------
# Training
# -------------------------

def train_and_save(train_dir: Path, model_path: Path):
    pdf_dir = train_dir / "pdfs"
    gt_dir = train_dir / "outputs"
    X, y = [], []

    log_debug(f"Training with data from {train_dir}")

    pdfs = sorted(p for p in pdf_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    for pdf_path in pdfs:
        gt_path = gt_dir / (pdf_path.stem + ".json")
        if not gt_path.exists():
            continue

        with open(gt_path) as f:
            gt_data = json.load(f)
        gt_headings = {(h["text"], h["page"]) for h in gt_data.get("outline", [])}

        blocks = extract_blocks(pdf_path)
        blocks = remove_repeating_headers_footers(blocks)
        df = extract_features(blocks)
        if df.empty:
            continue

        df["is_heading"] = df.apply(lambda r: (r["text"], r["page"]) in gt_headings, axis=1)
        X.append(df[FEATURES])
        y.append(df["is_heading"].astype(int))

    if not X:
        raise ValueError("No training data found")

    X_all = pd.concat(X)
    y_all = pd.concat(y)

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        GradientBoostingClassifier(max_depth=3, n_estimators=200, random_state=42),
    )
    model.fit(X_all, y_all)
    train_acc = model.score(X_all, y_all)

    model_data = {
        "model": model,
        "feature_names": FEATURES,
        "feature_order": list(X_all.columns),
    }
    joblib.dump(model_data, model_path)
    print(f"Trained model saved to {model_path} (samples: {len(y_all)}, accuracy: {train_acc:.4f})")


# -------------------------
# CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="PDF Outline Extractor")
    parser.add_argument("--input_dir", type=Path, help="Input PDF directory")
    parser.add_argument("--output_dir", type=Path, help="Output JSON directory")
    parser.add_argument("--model", type=Path, help="Path to trained model")
    parser.add_argument("--train_dir", type=Path, help="Training data directory")
    parser.add_argument("--one_based", action="store_true", help="Use 1-based page numbering")
    args = parser.parse_args()

    # Train
    if args.train_dir:
        if not args.model:
            print("Error: --model argument is required for training")
            return
        train_and_save(args.train_dir, args.model)
        return

    # Load model if present
    model_data = None
    if args.model and args.model.exists():
        try:
            model_data = joblib.load(args.model)
            print(f"Loaded model from {args.model}")
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)

    # Process PDFs
    if args.input_dir and args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pdf_paths = sorted(p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
        for pdf_path in pdf_paths:
            print(f"Processing {pdf_path.name}")
            result = process_file(pdf_path, model_data, args.one_based)

            # Validate schema if present
            if SCHEMA_PATH.exists():
                try:
                    with open(SCHEMA_PATH) as f:
                        schema = json.load(f)
                    validate(instance=result, schema=schema)
                except Exception as e:
                    print(f"Validation error for {pdf_path.name}: {e}", file=sys.stderr)

            output_path = args.output_dir / f"{pdf_path.stem}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            print(f"Saved {output_path} with {len(result['outline'])} outline items")


if __name__ == "__main__":
    main()