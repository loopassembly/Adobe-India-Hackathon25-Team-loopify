#!/usr/bin/env python3
"""PDF outline extractor for Adobe India Hackathon (Challenge 1-A).

One-pass pure-Python (pdfplumber) - < 10 s for a 50-page doc.
Hybrid: heuristic fallback + optional 50 KB LogisticRegression model.
Self-training: --train_dir with labelled pdf/json pairs to model.
Robust to noisy forms (bullet lists, field labels) - stricter pre-filters.
Zero or one-based page index via --one_based flag.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import List

import joblib
import pandas as pd
import pdfplumber
from jsonschema import validate
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCHEMA_PATH = Path(__file__).parent / "sample_dataset" / "schema" / "output_schema.json"

MIN_WORDS = 2
MIN_AVG_CHARS = 3
BULLET_CHARS = {"•", "·", "▪", "–", "-"}

def _looks_like_heading(txt: str) -> bool:
    if not txt or not txt.strip():
        return False
    txt = txt.strip()

    if re.search(r"[.··]{3,}|[--]{3,}|\s+\d+\s*$", txt):
        return False

    words = re.split(r"\s+", txt)
    if len(words) < MIN_WORDS or sum(map(len, words)) / len(words) < MIN_AVG_CHARS:
        return False
    if txt[0] in BULLET_CHARS or txt.rstrip().dswith((",", ":", ";")):
        return False
    if txt.isupper() and len(txt) < 15:
        return False

    if re.fullmatch(r"[\d\W]+", txt):
        return False

    return True

def _upper_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    uppers = sum(1 for c in letters if c.isupper())
    return uppers / len(letters)

def _title_case(text: str) -> int:
    words = re.findall(r"[A-Za-z]+", text)
    return int(bool(words) and all(w[0].isupper() for w in words))

def _looks_like_noise(text: str) -> bool:
    if text is None:
        return True
    try:
        t = text.strip()
        return bool(
            len(t) <= 2
            or re.fullmatch(r"[\d.·•–—-]+", t or "")
            or re.fullmatch(r"[.]{3,}", t or "")
        )
    except:
        return True

def extract_lines(pdf_path: Path) -> List[dict]:
    """Extract every line (pdfplumber words grouped by y) with font attrs."""
    lines: list[dict] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for pno, page in enumerate(pdf.pages):
            words = page.extract_words(
                x_tolerance=1,
                y_tolerance=1,
                keep_blank_chars=False,
                extra_attrs=["size", "fontname"]
            )
            line_map = {}
            for w in words:
                top = round(w["top"], 1)
                line_map.setdefault(top, []).append(w)

            sorted_tops = sorted(line_map.keys())
            last_bottom = 0

            for top in sorted_tops:
                line_words = line_map[top]
                text = " ".join(w["text"] for w in sorted(line_words, key=lambda d: d["x0"])).strip()
                if not _looks_like_heading(text):
                    continue
                font_sizes = [w.get("size", w.get("height", 0)) for w in line_words]
                if not font_sizes:
                    continue
                size = statistics.median(font_sizes)
                fontname = line_words[0].get("fontname", "")
                bold = bool(re.search(r"Bold|Semibold|Black", fontname, re.I))
                x0 = min(w["x0"] for w in line_words)
                x1 = max(w["x1"] for w in line_words)
                y0 = top
                y1 = top + size
                gap_above = top - last_bottom if last_bottom > 0 else 0
                page_width = page.width or 595
                centred = abs((page_width/2) - ((x0+x1)/2)) < page_width*0.15
                lines.append({
                    "text": text.strip(),
                    "size": size,
                    "bold": bold,
                    "centred": centred,
                    "x0": x0,
                    "x1": x1,
                    "y0": y0,
                    "gap_above": gap_above,
                    "line_height": size,
                    "page": pno,
                })
                last_bottom = y1
    return lines

def extract_features(pdf_path: Path) -> pd.DataFrame:
    lines = extract_lines(pdf_path)
    if not lines:
        return pd.DataFrame()
    df = pd.DataFrame(lines)
    body_medians = df.groupby("page")["size"].median().rename("body")
    df = df.join(body_medians, on="page")
    df["font_rel"] = df["size"] / df["body"].clip(lower=1e-3)
    df["gap_norm"] = df["gap_above"] / df["line_height"].clip(lower=1e-3)
    df["n_chars"] = df["text"].str.len()
    df["n_words"] = df["text"].str.split().str.len()
    df["upper_ratio"] = df["text"].apply(_upper_ratio)
    df["title_case"] = df["text"].apply(_title_case)
    df["noise"] = df["text"].apply(lambda x: 1 if _looks_like_noise(x) else 0)
    return df

FEATURES = [
    "font_rel",
    "gap_norm",
    "bold",
    "centred",
    "n_words",
    "upper_ratio",
    "title_case",
]

def load_training_samples(train_dir: Path):
    pdf_dir = train_dir / "pdfs"
    gt_dir = train_dir / "outputs"
    X, y = [], []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        gt_path = gt_dir / (pdf_path.stem + ".json")
        if not gt_path.exists():
            continue
        gt_headings = {(h["text"], h["page"]) for h in json.loads(gt_path.read_text()).get("outline", [])}
        df = extract_features(pdf_path)
        if df.empty:
            continue
        df["is_heading"] = df.apply(lambda r: (r["text"], r["page"]) in gt_headings, axis=1)
        X.append(df[FEATURES])
        y.append(df["is_heading"].astype(int))
    if not X:
        raise ValueError("No training data found - check --train_dir structure")
    return pd.concat(X), pd.concat(y)

def train_and_save(train_dir: Path, model_path: Path):
    X, y = load_training_samples(train_dir)
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        GradientBoostingClassifier(max_depth=3, n_estimators=200, random_state=42),
    )
    pipe.fit(X, y)
    joblib.dump(pipe, model_path)
    print(f"Trained model saved to {model_path} (samples: {len(y)})")

def heuristic_headings(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["noise"] == 0].copy()

    candidate = (
        ((df["font_rel"] >= 1.10) & (df["gap_norm"] >= 1.2))
        |
        (df["bold"] & df["centred"] & (df["n_words"] >= 3))
    )
    cand_df = df[candidate & (df["noise"] == 0)].copy()
    if cand_df.empty:
        return cand_df

    result_dfs = []
    for page_num in cand_df["page"].unique():
        page_df = cand_df[cand_df["page"] == page_num].copy()

        font_threshold = page_df["font_rel"].quantile(0.75)
        page_df = page_df[page_df["font_rel"] >= font_threshold]

        good_spacing = df[(df["page"] == page_num) & (df["gap_norm"] >= 2.0) & (df["font_rel"] >= 1.05)]
        page_df = pd.concat([page_df, good_spacing]).drop_duplicates()

        result_dfs.append(page_df)

    return pd.concat(result_dfs) if result_dfs else pd.DataFrame()

def assign_levels(headings: pd.DataFrame) -> pd.DataFrame:
    headings["size_rank"] = headings.groupby("page")["size"].rank("dense", ascending=False)
    headings["level"] = "H" + headings["size_rank"].astype(int).astype(str)
    return headings.drop(columns=["size_rank"])

def pick_title(df: pd.DataFrame) -> str:
    page1 = df[df["page"] == 0]
    if page1.empty:
        return ""
    biggest = page1.loc[page1["size"].idxmax()]
    if len(biggest["text"].split()) <= 10:
        return biggest["text"].strip()
    return ""

def clean_outline(outline):
    """Clean up outline by removing TOC artifacts and duplicates"""
    cleaned = []
    seen_texts = set()

    for h in outline:
        text = h["text"].strip()

        if re.search(r'[.]{3,}|[-]{3,}|\s+\d+\s*$', text):
            continue

        if len(text) < 3:
            continue

        normalized_text = re.sub(r'\s+', ' ', text.lower().strip())

        if normalized_text in seen_texts:
            continue

        seen_texts.add(normalized_text)
        cleaned.append(h)

    return cleaned

def process_file(pdf_path: Path, model=None, one_based: bool = False) -> dict:
    df = extract_features(pdf_path)

    if model is not None:
        cand = df.copy()
        proba = model.predict_proba(cand[FEATURES])[:,1]
        ml_headings = cand[proba > 0.25].copy()
        if len(ml_headings) < 3:
            headings = heuristic_headings(df)
        else:
            headings = ml_headings
    else:
        headings = heuristic_headings(df)

    if headings.empty:
        outline = []
    else:
        MAX_KEEP_PER_PAGE = 25
        def keep_top_per_page(group):
            return group.nlargest(MAX_KEEP_PER_PAGE, "size")

        headings = headings.groupby("page", group_keys=False).apply(keep_top_per_page)

        headings = assign_levels(headings)
        outline = headings.sort_values(["page", "y0"], ascending=[True, True])[
            ["level", "text", "page"]
        ].to_dict("records")
        outline = clean_outline(outline)
        if one_based:
            for h in outline:
                h["page"] += 1
    result = {
        "title": pick_title(df),
        "outline": outline,
    }
    return result

def main():
    p = argparse.ArgumentParser(description="Extract outline headings from PDFs.")
    p.add_argument("--input_dir", type=Path, help="Directory with input PDFs")
    p.add_argument("--output_dir", type=Path, help="Where JSONs will be written")
    p.add_argument("--model", type=Path, help="Path to heading_model.pkl (optional)")
    p.add_argument("--train_dir", type=Path, help="Labelled dataset (pdfs + outputs) to train model")
    p.add_argument("--one_based", action="store_true", help="Write pages starting at 1")
    args = p.parse_args()

    if args.train_dir:
        train_and_save(args.train_dir, args.model or Path("heading_model.pkl"))
        return

    model = None
    if args.model and args.model.exists():
        model = joblib.load(args.model)
        print(f"Using trained model from {args.model}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in sorted(args.input_dir.glob("*.pdf")):
        print(f"Processing {pdf_path.name}")
        result = process_file(pdf_path, model=model, one_based=args.one_based)
        schema = json.loads(SCHEMA_PATH.read_text())
        validate(instance=result, schema=schema)
        out_path = args.output_dir / (pdf_path.stem + ".json")
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Written {out_path}")

if __name__ == "__main__":
    main()
