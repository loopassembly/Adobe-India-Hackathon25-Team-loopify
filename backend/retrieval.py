# retrieval.py
import json, os, math
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

IDX_DIR = Path("data/index")
IDX_DIR.mkdir(parents=True, exist_ok=True)
FAISS_PATH = IDX_DIR / "sections.faiss"
META_PATH = IDX_DIR / "sections_meta.json"

# Default high-quality SBERT; ~420MB, great accuracy under 20GB image budget
EMB_MODEL_NAME = os.getenv("EMB_MODEL", "sentence-transformers/all-mpnet-base-v2")
_model = None
_reranker = None
_index = None
_meta: List[Dict[str, Any]] = []

def _get_encoder():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMB_MODEL_NAME, device="cpu")
    return _model

def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        except Exception:
            _reranker = None
    return _reranker

def _embed(texts: List[str]) -> np.ndarray:
    enc = _get_encoder()
    vecs = enc.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")

def _ensure_index():
    global _index, _meta
    if _index is not None:
        return
    if FAISS_PATH.exists() and META_PATH.exists():
        _index = faiss.read_index(str(FAISS_PATH))
        _meta = json.loads(META_PATH.read_text())
    else:
        _index = faiss.IndexFlatIP(768)  # cosine if vectors are normalized
        _meta = []

def _save_index():
    faiss.write_index(_index, str(FAISS_PATH))
    META_PATH.write_text(json.dumps(_meta, ensure_ascii=False))

# --- Sectionizer: use outline to carve sections and extract text ---
def _extract_text(p: pdfplumber.page.Page) -> str:
    return (p.extract_text() or "").strip()

def _to_sentences(text: str) -> List[str]:
    # very lightweight splitter
    parts = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return parts

def _mk_snippet(text: str) -> str:
    sents = _to_sentences(text)
    if len(sents) >= 3:
        return ". ".join(sents[:3]).strip() + "."
    elif sents:
        return ". ".join(sents[:2]).strip() + ("" if text.endswith(".") else ".")
    return text[:220]

def build_sections(pdf_path: Path, outline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not outline:
        return items
    with pdfplumber.open(str(pdf_path)) as pdf:
        n = len(pdf.pages)
        # sort by page asc
        ol = sorted(outline, key=lambda x: (x["page"], x["level"], x["text"]))
        for i, h in enumerate(ol):
            p0 = max(0, int(h["page"]))
            p1 = (int(ol[i + 1]["page"]) - 1) if (i + 1 < len(ol)) else (n - 1)
            p1 = max(p0, min(p1, n - 1))

            # concat text of pages p0..p1
            body = []
            for p in range(p0, p1 + 1):
                body.append(_extract_text(pdf.pages[p]))
            text = " ".join([t for t in body if t]).strip()

            items.append({
                "document": pdf_path.name,
                "section_title": h["text"],
                "level": h["level"],
                "page_number": p0,
                "end_page": p1,
                "content": text,
                "snippet": _mk_snippet(text) if text else "",
            })
    return items

# --- Public API ---
def add_document(pdf_path: Path, outline: List[Dict[str, Any]]):
    _ensure_index()
    sections = build_sections(pdf_path, outline)
    if not sections:
        return 0
    texts = []
    for s in sections:
        q = f"{s['section_title']}\n\n{s['content'][:1500]}"
        texts.append(q)
    vecs = _embed(texts)
    ids = list(range(len(_meta), len(_meta) + len(sections)))
    if _index.ntotal == 0:
        _index.add(vecs)
    else:
        _index.add(vecs)
    _meta.extend(sections)
    _save_index()
    return len(sections)

def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    _ensure_index()
    if _index.ntotal == 0:
        return []
    qv = _embed([query])
    D, I = _index.search(qv, min(top_k * 8, max(8, top_k * 8)))
    cand = [(_meta[int(i)], float(d)) for i, d in zip(I[0], D[0]) if int(i) < len(_meta)]
    # optional rerank
    reranker = _get_reranker()
    if reranker and cand:
        pairs = [(query, f"{c['section_title']}\n\n{c.get('content','')[:2000]}") for c, _ in cand]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(cand, scores), key=lambda x: x[1], reverse=True)[:top_k]
        out = []
        for (c, base_sim), rr in ranked:
            out.append({
                "document": c["document"],
                "section_title": c["section_title"],
                "page_number": c["page_number"],
                "score": float(rr),
                "snippet": c.get("snippet") or "",
            })
        return out
    # otherwise take top_k by embedding score
    out = []
    for c, d in cand[:top_k]:
        out.append({
            "document": c["document"],
            "section_title": c["section_title"],
            "page_number": c["page_number"],
            "score": float(d),
            "snippet": c.get("snippet") or "",
        })
    return out