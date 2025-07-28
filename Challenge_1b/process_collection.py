#!/usr/bin/env python3
"""Collection-level processor for Adobe India Hackathon Challenge 1-B

Given an input JSON and a folder of PDFs, this script selects the 5 most 
relevant top-level sections for the user's persona and task, then produces 
the summary JSON expected by the judges.

The script re-uses the heading-extraction utilities from Challenge 1-A
(process_pdfs.py). Uses TF-IDF and cosine similarity for offline processing.

Usage:
    python process_collection.py \
           --input_json "Collection 1/challenge1b_input.json" \
           --pdf_dir    "Collection 1/PDFs" \
           --output_json "Collection 1/my_output.json"

Dependencies:
    scikit-learn, pandas, numpy, pdfplumber, joblib
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
A1_DIR = ROOT_DIR / "Challenge_1a"
if str(A1_DIR) not in sys.path:
    sys.path.append(str(A1_DIR))

from process_pdfs import extract_features, heuristic_headings, assign_levels

WORD_LIMIT = 120
REFINED_LIMIT = 150
TOP_K_HEADINGS = 5


def _clean_text(t: str) -> str:
    """Collapse whitespace and strip bullets and footnote numbers."""
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^[-•·]\s*", "", t)
    t = re.sub(r"\s*\d+$", "", t)
    return t


def collect_candidate_chunks(pdf_dir: Path) -> List[Dict[str, Any]]:
    """Return list of candidate sections with their text content."""
    candidates: List[Dict[str, Any]] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        df = extract_features(pdf_path)
        if df.empty:
            continue
        heads = heuristic_headings(df)
        if heads.empty:
            continue
        heads = assign_levels(heads)
        heads = heads.sort_values(["page", "y0"]).reset_index(drop=True)

        for idx, row in heads.iterrows():
            page_num = int(row["page"])
            start_y = row["y0"]
            if idx + 1 < len(heads) and heads.loc[idx + 1, "page"] == page_num:
                end_y = heads.loc[idx + 1, "y0"]
            else:
                end_y = math.inf

            body_lines = (
                df[(df["page"] == page_num) & (df["y0"] > start_y) & (df["y0"] < end_y)]
                .sort_values("y0")
            )
            body_text = " ".join(body_lines["text"].tolist())
            body_text = _clean_text(body_text)
            if not body_text:
                continue

            candidate_text = f"{row['text']} {body_text}"
            candidates.append(
                {
                    "document": pdf_path.name,
                    "page": page_num,
                    "section_title": row["text"],
                    "text_for_rank": " ".join(candidate_text.split()[:WORD_LIMIT]),
                    "body_text": body_text,
                }
            )
    return candidates


def rank_sections(cands: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Return top-K candidate sections ranked by cosine similarity."""
    if not cands:
        return []
    corpus = [c["text_for_rank"] for c in cands]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=8000)
    X = vectorizer.fit_transform(corpus + [query])
    query_vec = X[-1]
    doc_mat = X[:-1]
    sims = cosine_similarity(doc_mat, query_vec)
    for c, s in zip(cands, sims.flatten()):
        c["similarity"] = float(s)
    top = sorted(cands, key=lambda d: d["similarity"], reverse=True)[:TOP_K_HEADINGS]
    for rank, c in enumerate(top, 1):
        c["importance_rank"] = rank
    return top


def refine_text(body: str) -> str:
    words = body.split()
    if len(words) > REFINED_LIMIT:
        body = " ".join(words[:REFINED_LIMIT]) + "..."
    return body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", type=Path, required=True, help="path to challenge1b_input.json")
    ap.add_argument("--pdf_dir", type=Path, required=True, help="directory containing the PDFs")
    ap.add_argument("--output_json", type=Path, required=True, help="where to write the result JSON")
    args = ap.parse_args()

    meta_in = json.loads(args.input_json.read_text())
    persona_obj = meta_in.get("persona", {})
    job_obj = meta_in.get("job_to_be_done", {})

    if isinstance(persona_obj, dict):
        persona = persona_obj.get("role", "")
    else:
        persona = str(persona_obj)

    if isinstance(job_obj, dict):
        job = job_obj.get("task", "")
    else:
        job = str(job_obj)

    query = f"{persona}. {job}"

    print("Collecting heading chunks...")
    candidates = collect_candidate_chunks(args.pdf_dir)
    print(f"Found {len(candidates)} candidate sections")

    ranked = rank_sections(candidates, query)

    extracted_sections = [
        {
            "document": c["document"],
            "section_title": c["section_title"],
            "importance_rank": c["importance_rank"],
            "page_number": c["page"],
        }
        for c in ranked
    ]

    subsection_analysis = [
        {
            "document": c["document"],
            "refined_text": refine_text(c["body_text"]),
            "page_number": c["page"],
        }
        for c in ranked
    ]

    input_docs = []
    if "documents" in meta_in:
        input_docs = [doc.get("filename", "") for doc in meta_in["documents"]]
    else:
        input_docs = meta_in.get("input_documents", [])

    output = {
        "metadata": {
            "input_documents": input_docs,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis,
    }

    args.output_json.write_text(json.dumps(output, indent=2))
    print(f"Written {args.output_json} with {len(extracted_sections)} sections")


if __name__ == "__main__":
    main()
