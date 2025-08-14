#!/usr/bin/env python3
"""Robust PDF Title/Outline Extractor (headers/footers aware, form-aware, multi-line titles)"""

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
import pdfplumber
import statistics
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
GLOBAL_SIZE_BUMP = 1.20  # for bold absolute-size rescue
ABS_SIZE_MIN = 14.0       # bold absolute-size rescue

# -------------------------
# Debug logger
# -------------------------
def log_debug(msg: str):
    if DEBUG:
        print(f"[DEBUG] {msg}", file=sys.stderr)

# -------------------------
# Text utilities
# -------------------------
def normalize_text(t: str) -> str:
    """Normalize obvious OCR/scan noise."""
    if not t:
        return t
    # collapse runs of same char >2 into a single char
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
    """Detect likely sentences (to exclude from headings)."""
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
    # drop pure ellipses/repeat dashes or trailing page numbers
    if re.search(r'[.]{3,}|[-]{3,}|\s+\d+\s*$', txt):
        return False
    # very short all-caps likely noise
    if txt.isupper() and len(txt) < 15:
        return False
    # drop sentence-like and lines ending with a period (allow !/?)
    if is_sentence_like(txt):
        return False
    if txt.rstrip().endswith('.'):
        return False
    return True

# -------------------------
# Line segmentation (split wide rows into segments)
# -------------------------
def split_line_segments(line_words: List[Dict[str, Any]], gap_multiplier: float = 1.8):
    """Split a visual text row into segments by large horizontal gaps."""
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
                    # split by big gaps -> take leftmost chunk as "line"
                    segments = split_line_segments(line_words, 1.8)
                    seg = sorted(segments, key=lambda s: min(w["x0"] for w in s))[0]

                    text = " ".join(w["text"] for w in sorted(seg, key=lambda d: d["x0"])).strip()
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
    """
    Remove only occurrences of (text, y_bin) that repeat across many pages.
    Leaves unique occurrences (e.g., title on page 0) intact.
    """
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
# Title extraction
# -------------------------
def extract_primary_title(blocks: List[Dict[str, Any]]) -> str:
    first = [b for b in blocks if b["page"] == 0]
    if not first:
        return ""
    title_block = max(first, key=lambda x: x["size"])
    return title_block["text"].strip()

def extract_composite_title(blocks: List[Dict[str, Any]]) -> str:
    """Join the largest centered line(s) on page 0 and optionally the immediate centered lines below."""
    first = [b for b in blocks if b["page"] == 0]
    if not first:
        return ""

    max_size = max(b["size"] for b in first)
    base = sorted(
        [b for b in first if (max_size - b["size"] <= 0.75) and (b["centred"] or b["y0"] < min(bb["y0"] for bb in first) + 200)],
        key=lambda b: (b["y0"], b["x0"]),
    )
    if not base:
        base = [max(first, key=lambda x: x["size"])]

    y_last = max(b["y0"] for b in base)
    # Extend with next centered-ish lines slightly smaller but still "titley"
    extensions = [b for b in first if (b["y0"] > y_last) and (b["y0"] <= y_last + 160) and (max_size - b["size"] <= 12)]
    parts: List[str] = []
    seen = set()
    for b in sorted(base + extensions, key=lambda x: (x["y0"], x["x0"])):
        if b["text"] not in seen:
            parts.append(b["text"])
            seen.add(b["text"])

    title = "  ".join(parts).strip()
    # Poster-ish title? return empty
    if upper_ratio(title) > 0.8 and len(title.split()) <= 6:
        return ""
    return title

def is_single_page(pdf_path: Path) -> bool:
    with pdfplumber.open(str(pdf_path)) as pdf:
        return len(pdf.pages) == 1

# -------------------------
# Form detection
# -------------------------
def is_form_document(blocks: List[Dict[str, Any]]) -> bool:
    if not blocks:
        return False
    texts = [b["text"] for b in blocks if b["page"] == 0]
    count_numbered = sum(1 for t in texts if re.match(r'^\d+(\.| )', t))
    form_keywords = [
        "S.No", "Signature of", "Date", "Designation", "Home Town",
        "Amount of advance", "PAY +", "Service Book"
    ]
    kw_hit = any(any(k.lower() in t.lower() for k in form_keywords) for t in texts)
    sizes = [b["size"] for b in blocks]
    std = statistics.pstdev(sizes) if len(sizes) > 1 else 0.0
    return (count_numbered >= 5 and kw_hit) or (kw_hit and std < 1.5 and len({b["page"] for b in blocks}) == 1)

# -------------------------
# Level assignment (global)
# -------------------------
def assign_levels_global(headings_df: pd.DataFrame) -> pd.DataFrame:
    df = headings_df.copy()
    df["size_bin"] = df["size"].round(1)
    uniq = sorted(df["size_bin"].unique(), reverse=True)
    level_map = {s: f"H{min(i + 1, 6)}" for i, s in enumerate(uniq)}
    df["level"] = df["size_bin"].map(level_map)
    return df.drop(columns=["size_bin"])

# -------------------------
# Optional OCR fallback (disabled by default unless env supports it)
# -------------------------
try:
    import pytesseract  # requires Tesseract installed
    from PIL import Image
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
# Feature extraction (for ML compat)
# -------------------------
def extract_features(blocks: List[Dict[str, Any]]) -> pd.DataFrame:
    if not blocks:
        return pd.DataFrame()
    df = pd.DataFrame(blocks).sort_values(["page", "y0"])
    # per-page body size
    body_medians = df.groupby("page")["size"].median().rename("body")
    df = df.join(body_medians, on="page")
    df["font_rel"] = df["size"] / df["body"].clip(lower=1e-3)
    df["n_chars"] = df["text"].str.len()
    df["n_words"] = df["text"].str.split().str.len()
    df["upper_ratio"] = df["text"].apply(upper_ratio)
    df["title_case"] = df["text"].apply(lambda x: int(is_title_case(x)))
    # vertical gap normalized by size
    df["gap"] = df.groupby("page")["y0"].diff().fillna(0.0)
    df["gap_norm"] = df["gap"] / df["size"].clip(lower=1e-3)
    return df

FEATURES = ["font_rel", "bold", "centred", "n_words", "upper_ratio", "title_case", "gap_norm"]

# -------------------------
# Helper: safe access to text column (fixes KeyError 'text')
# -------------------------
def _get_text_series(df_like: pd.DataFrame, fallback_df: pd.DataFrame) -> pd.Series:
    if "text" in df_like.columns:
        return df_like["text"].astype(str)
    # recover aligned text from original feature frame
    return fallback_df.loc[df_like.index, "text"].astype(str)

# -------------------------
# Core processing
# -------------------------
def process_file(pdf_path: Path, model_data=None, one_based: bool = False) -> Dict[str, Any]:
    # 1) Extract text blocks & clean headers/footers
    blocks = extract_blocks(pdf_path)
    blocks = remove_repeating_headers_footers(blocks)

    # Early out
    if not blocks:
        return {"title": "", "outline": []}

    # 2) Decide title
    single = is_single_page(pdf_path)
    if single:
        # posters/flyers often: leave title empty
        top = max([b for b in blocks if b["page"] == 0], key=lambda x: x["size"])
        posterish = (upper_ratio(top["text"]) > 0.7 and len(top["text"].split()) <= 8) or any(
            k in top["text"].upper() for k in ["INVITATION", "PARTY", "CONCERT", "SALE"]
        )
        title = "" if posterish else top["text"]
    else:
        # combine multi-line title on first page
        title = extract_composite_title(blocks) or extract_primary_title(blocks)

    # 3) Form documents → only title, no outline
    if is_form_document(blocks):
        return {"title": extract_primary_title(blocks), "outline": []}

    # 4) Prepare features
    df = extract_features(blocks)
    if df.empty:
        return {"title": title, "outline": []}

    global_median_size = df["size"].median()

    # 5) Predict via model if provided
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

    # ---- robust filtering using safe text series (fixes KeyError 'text') ----
    txt = _get_text_series(cand, df)
    cand = cand[~txt.apply(is_sentence_like)]

    txt = _get_text_series(cand, df)  # refresh after filter
    cand = cand[~txt.str.strip().str.endswith(".")]

    # drop title components from outline on first page
    if title:
        title_parts = [p.strip() for p in title.split("  ")]
        txt = _get_text_series(cand, df)
        cand = cand[~((cand["page"] == 0) & (txt.isin(title_parts)))]

    # single-page flyers: keep only strongest top-level heading on the page
    if single and not cand.empty:
        p0 = cand[cand["page"] == 0].copy()
        if not p0.empty:
            page_max = p0["size"].max()
            near = p0[p0["size"] >= page_max - 0.6]
            keep_idx = near.sort_values("y0").head(1).index
            cand = cand.drop(index=p0.index.difference(keep_idx))

    if cand.empty:
        outline = []
    else:
        # Ensure 'text' exists before drop_duplicates
        txt = _get_text_series(cand, df)
        cand = cand.assign(text=txt.values)
        cand = assign_levels_global(cand.drop_duplicates(subset=["text", "page"]))
        outline = cand[["level", "text", "page"]].to_dict("records")

    # 7) Optional OCR fallback for single-page posters w/ no outline
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

    # page numbering option
    if one_based:
        for item in outline:
            item["page"] += 1

    return {"title": title, "outline": outline}

# -------------------------
# Training (unchanged API)
# -------------------------
def train_and_save(train_dir: Path, model_path: Path):
    pdf_dir = train_dir / "pdfs"
    gt_dir = train_dir / "outputs"
    X, y = [], []

    log_debug(f"Training with data from {train_dir}")

    # case-insensitive *.pdf scan
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
        # case-insensitive scan
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
