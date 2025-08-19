# server.py
import os
import asyncio
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional

import re
import time
import json, uuid
from collections import OrderedDict

from fastapi import FastAPI, UploadFile, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from process_pdfs import process_file
import retrieval  # we may reload this module when switching model names
from audio_bridge import synthesize
from chat_with_llm import get_llm_response  # <-- use the provided file directly
from scoring import coerce_score_percent

# ==============================================================================
# Paths & App
# ==============================================================================
APP_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_ROOT", APP_DIR / "data")).resolve()
PDF_DIR = DATA_DIR / "pdfs"
AUDIO_DIR = DATA_DIR / "audio"
INDEX_DIR = DATA_DIR / "index"
PDF_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# NOTE: move Swagger so our GET /docs can exist
app = FastAPI(title="DocDots Backend", version="3.7", docs_url="/swagger", redoc_url=None)

# (Starlette type import kept; harmless if unused)
from starlette.types import ASGIApp, Receive, Scope, Send

@app.middleware("http")
async def add_pdf_cors_and_ranges(request, call_next):
    response = await call_next(request)
    # Only decorate PDFs
    ctype = response.headers.get("content-type", "")
    if "application/pdf" in ctype:
        # allow Adobe iframe to fetch cross-origin
        response.headers.setdefault("Access-Control-Allow-Origin", "*")
        # range support for streaming/seeking
        response.headers.setdefault("Accept-Ranges", "bytes")
        # cache a bit to avoid re-fetches
        response.headers.setdefault("Cache-Control", "public, max-age=3600")
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten in prod if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static data (pdfs + audio)
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")

# In-memory registry: name -> { path: Path, title?: str, outline?: [...] }
DOCS: Dict[str, Dict[str, Any]] = {}

# Global status (for model warmup & general progress)
STATUS: Dict[str, Any] = {
    "phase": "idle",           # idle | checking-cache | downloading | loading | ready | error
    "progress": 0,             # 0..100
    "message": "",
    "model": None,
    "provider": os.getenv("LLM_PROVIDER", "gemini"),
}

# ==============================================================================
# Helpers
# ==============================================================================
def _set_status(phase: str, progress: int, message: str):
    STATUS["phase"] = phase
    STATUS["progress"] = max(0, min(100, progress))
    STATUS["message"] = message
    print(f"[STATUS] {phase} {STATUS['progress']}% — {message}")

def _register_on_disk() -> None:
    for p in sorted(PDF_DIR.glob("*.pdf")):
        DOCS.setdefault(p.name, {"path": p})

def _register_file(name: str) -> None:
    p = PDF_DIR / name
    if p.exists():
        DOCS.setdefault(name, {"path": p})

def _outline_for(name: str) -> Dict[str, Any]:
    if name not in DOCS:
        _register_file(name)
    if name not in DOCS:
        raise FileNotFoundError(f"Unknown document: {name}")
    if "outline" in DOCS[name]:
        return {"title": DOCS[name].get("title", ""), "outline": DOCS[name]["outline"]}
    pdf_path: Path = DOCS[name]["path"]
    out = process_file(pdf_path)
    DOCS[name]["title"] = out.get("title", "")
    DOCS[name]["outline"] = out.get("outline", [])
    return out

def _query_from_doc_page(name: str, page: int) -> str:
    o = _outline_for(name)
    outline = o.get("outline", [])
    if not outline:
        return f"{name} page {page+1}"
    # nearest heading at/before page, else closest
    best = None
    for h in outline:
        if h["page"] <= page and (best is None or h["page"] > best["page"]):
            best = h
    if best is None:
        best = min(outline, key=lambda x: abs(x["page"] - page))
    return f"{best['text']} (from {name})"

# ------------------------------------------------------------------------------
# Scoring & Snippet utilities
# ------------------------------------------------------------------------------
MIN_SCORE_PCT_DEFAULT = 80.0
TOP_K_DEFAULT = 5
MIN_SCORE_PCT = float(os.getenv("MIN_SCORE_PCT", str(MIN_SCORE_PCT_DEFAULT)))

# Strict gating & lexical overlap controls
RECO_STRICT = str(os.getenv("RECO_STRICT", "1")).lower() not in ("0", "false", "no")
MIN_OVERLAP_TOKENS = int(os.getenv("MIN_OVERLAP_TOKENS", "1"))

# ---------------- Performance & Caching knobs ----------------
QUERY_MAX_CHARS = int(os.getenv("QUERY_MAX_CHARS", "512"))           # cap query length before embedding
SEARCH_CACHE_TTL_SEC = int(os.getenv("SEARCH_CACHE_TTL_SEC", "300")) # 5 min TTL
SEARCH_CACHE_MAX = int(os.getenv("SEARCH_CACHE_MAX", "256"))         # LRU size
RECO_BUDGET_MS = int(os.getenv("RECO_BUDGET_MS", "1500"))            # total budget per request (ms)

# --- Podcast duration tuning (env-tunable) ------------------------------------
PODCAST_MIN_MINUTES = float(os.getenv("PODCAST_MIN_MINUTES", "3.0"))  # hard floor
PODCAST_MAX_MINUTES = float(os.getenv("PODCAST_MAX_MINUTES", "5.0"))  # hard ceiling
TTS_WPM = int(os.getenv("TTS_WPM", "200"))  # realistic for alloy/verse
SEGMENT_CHARS_MIN = int(os.getenv("PODCAST_SEGMENT_CHARS_MIN", "120"))
SEGMENT_CHARS_MAX = int(os.getenv("PODCAST_SEGMENT_CHARS_MAX", "260"))
EXPANSION_TOLERANCE = float(os.getenv("PODCAST_EXPANSION_TOL", "0.95"))  # expand if <95% of target words

# very small stopword list to avoid matching on meaningless tokens
_STOP = {
    "the","and","or","of","in","to","a","an","for","on","with","by","at","from","is","are","was","were",
}

def _tokenize(s: str) -> list:
    if not s:
        return []
    return [t for t in re.findall(r"[A-Za-z0-9]+", s.lower()) if len(t) > 2 and t not in _STOP]

def _token_overlap(q: str, t: str) -> int:
    qset = set(_tokenize(q))
    tset = set(_tokenize(t))
    return len(qset & tset)

# ---------------- Query prep + LRU search cache ----------------
SEARCH_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()  # key -> {"ts": float, "raw": list}
INDEX_EPOCH = 0  # bump whenever /index runs so cache is isolated per index build

_def_now = time.monotonic
def _now() -> float:
    return _def_now()

def _prepare_query_for_embedding(q: str) -> str:
    """Trim/normalize very long selections before embedding to reduce CPU & latency."""
    if not q:
        return ""
    s = re.sub(r"\s+", " ", q).strip()
    if len(s) <= QUERY_MAX_CHARS:
        return s
    # keep head+tail to preserve key entities while bounding length
    head = s[: QUERY_MAX_CHARS // 2]
    tail = s[-QUERY_MAX_CHARS // 2 :]
    return f"{head} … {tail}"

def _cache_key(epoch: int, query: str, topn: int) -> tuple:
    return (epoch, query, int(topn))

def _cache_prune():
    # expire by TTL
    now = _now()
    expired = []
    for k, v in list(SEARCH_CACHE.items()):
        if now - v.get("ts", 0.0) > SEARCH_CACHE_TTL_SEC:
            expired.append(k)
    for k in expired:
        SEARCH_CACHE.pop(k, None)
    # enforce size as LRU (most recent at end)
    while len(SEARCH_CACHE) > SEARCH_CACHE_MAX:
        SEARCH_CACHE.popitem(last=False)

def _search_raw_cached(query: str, topn: int):
    """Thin wrapper over retrieval.search with LRU+TTL caching and index epoch isolation."""
    k = _cache_key(INDEX_EPOCH, query, int(topn))
    item = SEARCH_CACHE.get(k)
    if item is not None:
        SEARCH_CACHE.move_to_end(k)  # LRU touch
        return item.get("raw")
    raw = retrieval.search(query, top_k=topn)
    SEARCH_CACHE[k] = {"ts": _now(), "raw": raw}
    _cache_prune()
    return raw

# Optional LLM re-ranking of related items (to boost semantic relevance)
# RECO_USE_LLM_RERANK: "0" to disable, "1" to always use, "auto" to trigger only when top1 < RECO_TRIGGER_UNDER.
RECO_USE_LLM_RERANK = os.getenv("RECO_USE_LLM_RERANK", "auto")
RECO_LLM_TOPN = int(os.getenv("RECO_LLM_TOPN", "8"))          # how many top items to re-rank with LLM
RECO_LLM_ALPHA = float(os.getenv("RECO_LLM_ALPHA", "0.35"))    # blend weight for LLM score [0..1]
RECO_TRIGGER_UNDER = float(os.getenv("RECO_TRIGGER_UNDER", "80"))

def _make_snippet(section_text: str, query: str, max_sentences: int = 4) -> str:
    """
    Produce a 2–4 sentence snippet around where the query best matches.
    Falls back to the first few sentences if no fuzzy match.
    """
    text = (section_text or "").strip()
    if not text:
        return ""
    # naive sentence split
    sents = re.split(r'(?<=[.!?])\s+', text)
    if not sents:
        return text[:300]

    q = (query or "").strip().lower()
    best_idx = 0
    if q:
        # score each sentence by token overlap
        qtoks = {t for t in re.findall(r"\w+", q) if len(t) > 2}
        best = -1
        for i, s in enumerate(sents):
            stoks = {t for t in re.findall(r"\w+", s.lower()) if len(t) > 2}
            overlap = len(qtoks & stoks)
            if overlap > best:
                best = overlap
                best_idx = i

    start = max(0, best_idx - 1)
    end = min(len(sents), start + max(2, min(max_sentences, 4)))
    snippet = " ".join(sents[start:end]).strip()
    # keep snippet sane length
    if len(snippet) > 480:
        snippet = snippet[:480].rsplit(" ", 1)[0] + "…"
    return snippet

def _normalize_related_items(
    raw_items,
    query: str,
    min_score_pct: float = MIN_SCORE_PCT,
    top_k: int = TOP_K_DEFAULT,
    *,
    strict: bool = RECO_STRICT,
    min_overlap_tokens: int = MIN_OVERLAP_TOKENS,
):
    """
    Convert retriever outputs to a clean shape with REAL scores while enforcing quality gates.
    - score_pct is derived from score_raw using scoring.coerce_score_percent (cosine/IP -> 0..100).
    - Discards candidates below min_score_pct.
    - Discards candidates with near-zero lexical overlap (helps avoid spurious matches like
      "Greek yogurt" vs. "Greek sailors").
    - If strict and none pass, returns an empty list (no fabricated top-Ks).
    """
    items = []
    raw_items = raw_items or []
    q = (query or "").strip()

    for it in raw_items:
        score_raw = it.get("score", it.get("similarity", 0.0))
        pct = coerce_score_percent(score_raw)

        # Prefer provided snippet; otherwise build one from the available text fields
        snippet = it.get("snippet")
        if not snippet:
            section_text = it.get("text") or it.get("section_text") or it.get("content") or ""
            snippet = _make_snippet(section_text, q)

        # Normalize and carry forward expected fields
        clean = {k: v for k, v in it.items() if k not in ("score", "score_pct", "score_raw")}
        if "page_number" in clean:
            try:
                clean["page_number"] = int(clean["page_number"])
            except Exception:
                pass

        # lightweight lexical sanity-check
        overlap = _token_overlap(q, f"{clean.get('section_title','')}\n{snippet}")

        merged = {
            **clean,
            "score_raw": float(score_raw),
            "score_pct": float(pct),
            "score": float(pct),  # the client-facing score; keep identical to pct
            "snippet": snippet,
            "overlap_tokens": int(overlap),
        }
        items.append(merged)

    # Hard filters: semantic threshold + lexical overlap gate
    filtered = [
        x for x in items
        if float(x.get("score_pct", 0.0)) >= float(min_score_pct)
        and int(x.get("overlap_tokens", 0)) >= int(min_overlap_tokens)
    ]

    # Sort by semantic score desc
    filtered.sort(key=lambda x: x.get("score_pct", 0.0), reverse=True)

    if not filtered:
        # In strict mode, don't return weak matches at all.
        return [] if strict else items[: int(top_k)]

    return filtered[: int(top_k)]

# ------------------------------------------------------------------------------
# LLM Re-ranking utilities
# ------------------------------------------------------------------------------
def _llm_rerank(query: str, items: List[Dict[str, Any]], *, timeout_s: Optional[float] = None) -> List[float]:
    """
    Ask the configured LLM to rate each candidate (0..100) for relevance to the query.
    Returns a list of scores aligned with items. Falls back to zeros on failure.
    """
    if not items:
        return []
    # Build a compact, deterministic prompt
    numbered = []
    for i, it in enumerate(items, 1):
        title = it.get("section_title", "")
        snippet = it.get("snippet", "") or it.get("content", "") or it.get("section_text", "")
        doc = it.get("document", "")
        page = int(it.get("page_number", 0)) + 1
        # Keep each candidate short to save tokens and time
        snippet = (snippet or "").strip()
        if len(snippet) > 400:
            snippet = snippet[:400].rsplit(" ", 1)[0] + "…"
        numbered.append(f"{i}. {title}  (doc: {doc}, p{page})\n   {snippet}")

    system = (
        "You are a re-ranker for a document reader. "
        "Rate how relevant each candidate section is to the user's query. "
        "Use ONLY the text provided. "
        "Return one line per item in the strict format 'i: SCORE' with i as the 1-based index and SCORE in 0..100. "
        "No explanations."
    )
    user = f"Query:\n{query}\n\nCandidates:\n" + "\n".join(numbered) + "\n\nRespond with lines like:\n1: 87\n2: 65\n..."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        text = (get_llm_response(messages) or "").strip()
    except Exception as e:
        print("[WARN] LLM rerank failed:", repr(e))
        return [0.0] * len(items)

    scores = [0.0] * len(items)
    for line in text.splitlines():
        m = re.search(r"^\s*(\d+)\s*[:\-]\s*(\d{1,3})\s*$", line.strip())
        if not m:
            continue
        idx = int(m.group(1)) - 1
        val = float(m.group(2))
        if 0 <= idx < len(items):
            scores[idx] = max(0.0, min(100.0, val))
    return scores

def _maybe_llm_rerank(
    query: str,
    items: List[Dict[str, Any]],
    *,
    top_k: int = TOP_K_DEFAULT,
    time_budget_left: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Optionally re-rank the top-N results using an LLM and blend with the base score.
    Controlled by env vars: RECO_USE_LLM_RERANK, RECO_LLM_TOPN, RECO_LLM_ALPHA, RECO_TRIGGER_UNDER.
    Skips if time budget is exhausted.
    """
    mode = str(RECO_USE_LLM_RERANK or "").lower()
    if mode not in ("1", "true", "auto"):
        return items
    if not items:
        return items
    # no time left -> skip
    if time_budget_left is not None and time_budget_left <= 0.02:
        return items
    if mode == "auto":
        top1 = float(items[0].get("score_pct", 0.0))
        if top1 >= float(RECO_TRIGGER_UNDER):
            return items

    topn = max(1, min(int(RECO_LLM_TOPN), len(items)))
    candidates = items[:topn]
    try:
        llm_scores = _llm_rerank(query, candidates, timeout_s=time_budget_left)
    except Exception:
        llm_scores = []
    if not llm_scores or not any(s > 0 for s in llm_scores):
        return items

    alpha = max(0.0, min(1.0, float(RECO_LLM_ALPHA)))
    for i, s in enumerate(llm_scores):
        base = float(candidates[i].get("score_pct", 0.0))
        blended = (1.0 - alpha) * base + alpha * float(s)
        candidates[i]["score_llm"] = float(s)
        candidates[i]["score_pct"] = float(blended)
        candidates[i]["score"] = float(blended)

    candidates.sort(key=lambda x: x.get("score_pct", 0.0), reverse=True)
    tail = items[topn:]
    merged = candidates + tail
    filtered = [x for x in merged if x.get("score_pct", 0.0) >= float(MIN_SCORE_PCT)]
    if RECO_STRICT and not filtered:
        return []
    return (filtered or merged)[: top_k]

def _prewarm_embeddings_blocking():
    requested = os.getenv("EMB_MODEL_NAME", "").strip() or None
    fallbacks = [
        requested,
        "Alibaba-NLP/gte-base-en-v1.5",
        "thenlper/gte-small",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]
    tried: List[str] = []
    for candidate in [m for m in fallbacks if m]:
        try:
            os.environ["EMB_MODEL_NAME"] = candidate
            _set_status("checking-cache", 5, f"Checking local cache for '{candidate}'…")
            importlib.reload(retrieval)

            _set_status("downloading", 10, f"Downloading model '{candidate}' (first run)…")
            if hasattr(retrieval, "prewarm"):
                retrieval.prewarm()  # type: ignore[attr-defined]
            else:
                try:
                    _ = retrieval.search("warmup", top_k=1)
                except Exception:
                    if hasattr(retrieval, "_get_encoder"):
                        retrieval._get_encoder()  # type: ignore[attr-defined]

            _set_status("loading", 80, f"Initializing '{candidate}'…")
            try:
                _ = retrieval.search("ok", top_k=1)
            except Exception:
                pass

            _set_status("ready", 100, f"Embeddings ready: {candidate}")
            STATUS["model"] = candidate
            return
        except Exception as e:
            tried.append(f"{candidate} -> {repr(e)}")
            _set_status("error", 0, f"Prewarm failed for '{candidate}', trying next…")

    STATUS["model"] = None
    msg = "All candidate embedding models failed. Tried:\n" + "\n".join(tried)
    _set_status("error", 0, msg)

async def _prewarm_embeddings_async():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _prewarm_embeddings_blocking)

# Register PDFs on disk at import time
_register_on_disk()

# ==============================================================================
# Request Models
# ==============================================================================
class RecoRequest(BaseModel):
    document: str
    page: int = 0
    selection: str = ""
    top_k: int = 5

class InsightsRequest(BaseModel):
    document: str
    page: int = 0
    selection: str = ""
    top_k: int = 3

class SelectRequest(BaseModel):
    selection: str
    top_k: int = 5

# ------------------------------------------------------------------------------
# Podcast/Audio Generation Request Model
# ------------------------------------------------------------------------------
class PodcastGenRequest(BaseModel):
    document: Optional[str] = None
    page: int = 0
    selection: str = ""
    style: str = "podcast"          # "podcast" (2 speakers) or "overview" (1 speaker)
    speakers: int = 2               # preferred number of speakers (1 or 2)
    duration_min: float = 3.0       # default target length (min 3 as per requirement)
    voices: Optional[List[str]] = None  # e.g., ["alloy","verse"] for Azure OpenAI TTS
    format: str = "audio-48khz-192kbitrate-mono-mp3"

# ==============================================================================
# Lifecycle
# ==============================================================================
@app.on_event("startup")
async def _on_startup():
    STATUS["provider"] = os.getenv("LLM_PROVIDER", "gemini")
    try:
        asyncio.create_task(_prewarm_embeddings_async())
    except Exception as e:
        _set_status("error", 0, f"Failed to schedule prewarm: {e!r}")

# ==============================================================================
# Endpoints
# ==============================================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "docs": len(DOCS),
        "phase": STATUS["phase"],
        "progress": STATUS["progress"],
        "message": STATUS["message"],
        "embedding_model": STATUS["model"],
        "min_score_pct": MIN_SCORE_PCT,
        "llm_provider": STATUS["provider"],
        "data_dir": str(DATA_DIR),
        "swagger": "/swagger",
        "search_cache_entries": len(SEARCH_CACHE),
        "index_epoch": INDEX_EPOCH,
        "query_max_chars": QUERY_MAX_CHARS,
        "reco_budget_ms": RECO_BUDGET_MS,
        # --- Extra non-secret TTS debug info
        "tts_provider": os.getenv("TTS_PROVIDER", ""),
        "azure_tts_endpoint": os.getenv("AZURE_TTS_ENDPOINT", ""),
        "default_voices": _default_voice_list(),
    }

@app.get("/status")
def status():
    return {
        "phase": STATUS["phase"],
        "progress": STATUS["progress"],
        "message": STATUS["message"],
        "embedding_model": STATUS["model"],
    }

@app.get("/tts/test")
def tts_test():
    try:
        prov = (os.getenv("TTS_PROVIDER", "azure") or "").lower()
        default_voice = (
            os.getenv("AZURE_TTS_VOICE", "alloy") if prov == "azure"
            else os.getenv("GCP_TTS_VOICE", "en-US-Neural2-F")
        )
        p = synthesize(
            "Hello from DocDots. This is a quick test.",
            voice=default_voice,
            fmt="audio-48khz-192kbitrate-mono-mp3",
        )
        if (not p.exists()) or p.stat().st_size == 0:
            return JSONResponse(
                {"error": "TTS produced no audio", "provider": prov, "voice_used": default_voice},
                status_code=502,
            )
        return {"audio_url": f"/data/audio/{p.name}", "bytes": p.stat().st_size, "provider": prov, "voice_used": default_voice}
    except Exception as e:
        return JSONResponse({"error": "TTS test failed", "detail": str(e), "provider": os.getenv("TTS_PROVIDER", "")}, status_code=500)

@app.get("/docs")  # <-- now safe; Swagger moved to /swagger
def list_docs():
    _register_on_disk()
    files = sorted(DOCS.keys())
    return {"docs": files}

@app.post("/index")
async def index_docs(request: Request):
    """
    Upload and/or (re)index PDFs.
    Accepts multipart with these keys: files | files[] | file | upload | pdf | documents
    Also works with an empty body to (re)index what's already on disk.
    """
    content_type = (request.headers.get("content-type") or "").lower()
    uploads: List[UploadFile] = []
    if "multipart/form-data" in content_type:
        try:
            form = await request.form()
            candidate_keys = ("files", "files[]", "file", "upload", "pdf", "documents")
            if hasattr(form, "getlist"):
                for key in candidate_keys:
                    for v in form.getlist(key):
                        if v and hasattr(v, "filename"):
                            uploads.append(v)
            if hasattr(form, "multi_items"):
                for _, v in form.multi_items():
                    if v and hasattr(v, "filename") and v not in uploads:
                        uploads.append(v)
        except Exception as e:
            print(f"[WARN] multipart parse failed (content-type={content_type!r}): {e!r}")

    added: List[str] = []
    for f in uploads:
        try:
            fn = getattr(f, "filename", "") or ""
            if not fn.lower().endswith(".pdf"):
                continue
            dst = PDF_DIR / fn
            content = await f.read()
            dst.write_bytes(content)
            _register_file(fn)
            added.append(fn)
        except Exception as e:
            print(f"[WARN] saving {getattr(f, 'filename', '<unknown>')} failed: {e}")
        finally:
            try:
                await f.close()
            except Exception:
                pass

    _register_on_disk()
    targets = added if added else sorted(DOCS.keys())
    built: List[str] = []
    for name in targets:
        try:
            out = _outline_for(name)
            retrieval.add_document(PDF_DIR / name, out.get("outline", []))
            built.append(name)
        except Exception as e:
            print(f"[WARN] index failed {name}: {e}")

    # Invalidate search cache after (re)indexing
    global INDEX_EPOCH
    INDEX_EPOCH += 1
    SEARCH_CACHE.clear()

    return {
        "status": "ok",
        "docs_added": added,
        "docs_built": built,
        "docs": sorted(DOCS.keys()),
        "content_type": content_type,
        "upload_count": len(uploads),
    }

@app.get("/outline")
def get_outline(document: str = Query(...)):
    try:
        o = _outline_for(document)
        return JSONResponse(o)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)

@app.post("/recommendations")
def recommendations(body: RecoRequest):
    t_start = _now()
    name = body.document
    page = int(body.page)
    top_k = int(body.top_k)
    selection = (body.selection or "").strip()

    query_full = selection or _query_from_doc_page(name, page)
    query = _prepare_query_for_embedding(query_full)

    raw = _search_raw_cached(query, topn=max(20, top_k))
    t_after_search = _now()

    results = _normalize_related_items(raw, query=query, min_score_pct=MIN_SCORE_PCT, top_k=top_k)
    # Optional LLM re-ranking under time budget
    elapsed_ms = int((_now() - t_start) * 1000)
    budget_left_s = max(0.0, (RECO_BUDGET_MS - elapsed_ms) / 1000.0)
    results = _maybe_llm_rerank(query, results, top_k=top_k, time_budget_left=budget_left_s)

    # Final strict gating for UI: only surface 80%+ items (and overlap gate already applied)
    if RECO_STRICT:
        results = [r for r in results if float(r.get("score", 0.0)) >= float(MIN_SCORE_PCT)]

    timings = {
        "search_ms": int((t_after_search - t_start) * 1000),
        "normalize_ms": int((_now() - t_after_search) * 1000),
        "total_ms": int((_now() - t_start) * 1000),
        "cache_entries": len(SEARCH_CACHE),
        "epoch": INDEX_EPOCH,
    }
    return {"results": results, "query_used": query, "rerank_mode": RECO_USE_LLM_RERANK, "timings": timings}

# ---------------- TTS voice helpers ----------------
_OPENAI_OAI_VOICES = {"alloy", "echo", "shimmer", "onyx", "fable", "nova", "coral", "verse", "ballad", "ash", "sage"}

def _default_tts_voice() -> str:
    """Pick a sensible default voice based on TTS_PROVIDER."""
    prov = (os.getenv("TTS_PROVIDER", "azure") or "azure").lower()
    if prov == "azure":
        # Azure OpenAI TTS uses OpenAI-style short names (e.g., alloy/echo/shimmer)
        return os.getenv("AZURE_TTS_VOICE", "alloy")
    elif prov == "gcp":
        return os.getenv("GCP_TTS_VOICE", "en-US-Neural2-F")
    else:
        return os.getenv("ESPEAK_VOICE", "en")

def _coerce_voice_for_provider(voice: Optional[str]) -> str:
    """If an Azure Speech voice like 'en-US-JennyNeural' slips through for Azure OpenAI, map to a valid one."""
    v = (voice or "").strip()
    prov = (os.getenv("TTS_PROVIDER", "azure") or "azure").lower()
    if prov == "azure":
        # Azure OpenAI only accepts OpenAI voices (e.g. alloy/echo/shimmer). If we detect a Speech-style voice, fall back.
        # Speech-style patterns look like 'en-US-*-Neural'.
        if not v or v.lower() not in _OPENAI_OAI_VOICES or "neural" in v.lower() or "-" in v:
            return os.getenv("AZURE_TTS_VOICE", "alloy")
        return v.lower()
    # For GCP/local, keep whatever was passed/defaulted.
    return v or _default_tts_voice()

# --- Provider-aware default voices + picker -----------------------------------
def _default_voice_list() -> List[str]:
    prov = (os.getenv("TTS_PROVIDER", "azure") or "").lower()
    if prov == "azure":
        # Azure OpenAI TTS voices (OpenAI-style)
        v1 = os.getenv("AZURE_TTS_VOICE", "alloy")
        v2 = os.getenv("PODCAST_VOICE_2", "verse")
        return [v1, v2]
    elif prov == "gcp":
        # Google TTS voices
        v1 = os.getenv("GCP_TTS_VOICE", "en-US-Neural2-F")
        v2 = os.getenv("PODCAST_VOICE_2", "en-US-Neural2-D")
        return [v1, v2]
    else:
        # Local/espeak defaults
        v1 = os.getenv("ESPEAK_VOICE", "en")
        v2 = os.getenv("ESPEAK_VOICE2", "en+f3")
        return [v1, v2]

def _pick_voices(speakers: int, voices: Optional[List[str]] = None) -> List[str]:
    prov = (os.getenv("TTS_PROVIDER", "azure") or "azure").lower()

    if voices and isinstance(voices, list) and all(isinstance(v, str) for v in voices):
        cand = [ _coerce_voice_for_provider(v) for v in voices if v ]
    else:
        if prov == "azure":
            # Azure OpenAI defaults (two speakers)
            cand = [
                _coerce_voice_for_provider(os.getenv("PODCAST_VOICE_1", os.getenv("AZURE_TTS_VOICE", "alloy"))),
                _coerce_voice_for_provider(os.getenv("PODCAST_VOICE_2", "verse")),
            ]
        elif prov == "gcp":
            cand = [
                os.getenv("PODCAST_VOICE_1", os.getenv("GCP_TTS_VOICE", "en-US-Neural2-F")),
                os.getenv("PODCAST_VOICE_2", "en-US-Neural2-A"),
            ]
        else:
            cand = [os.getenv("PODCAST_VOICE_1", "en"), os.getenv("PODCAST_VOICE_2", "en")]

    if not cand:
        cand = [_default_tts_voice(), _default_tts_voice()]
    # ensure at least 2 entries for alternating even if speakers==1 (we'll only use first)
    while len(cand) < 2:
        cand.append(cand[-1])

    return cand[: max(1, speakers)]

def _json_from_text(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    # try to extract the outermost JSON object from the text
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end+1])
    except Exception:
        return None
    return None

# --------- Podcast planning, sizing, and stitching ----------------------------

def _count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text or ""))

def _plan_word_count(plan: Dict[str, Any]) -> int:
    segs = (plan or {}).get("segments") or []
    return sum(_count_words((s.get("text") or "")) for s in segs)

def _build_podcast_messages(
    focus: str,
    ctx_items: List[Dict[str, Any]],
    *,
    style: str = "podcast",
    speakers: int = 2,
    duration_min: float = 3.0,
    insights_markdown: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Ask the LLM to produce a grounded podcast script as JSON.
    Uses the current section, related sections (ctx_items), and optional Insights markdown.
    """
    def _ctx(items: List[Dict[str, Any]], max_snip: int = 420) -> str:
        lines = []
        for i, it in enumerate(items, 1):
            title = (it.get("section_title") or "").strip()
            snippet = (it.get("snippet") or it.get("content") or it.get("section_text") or "").strip()
            if len(snippet) > max_snip:
                snippet = snippet[:max_snip].rsplit(" ", 1)[0] + "…"
            doc = (it.get("document") or "?")
            page = int(it.get("page_number", 0)) + 1
            lines.append(f"[{i}] {title} — {snippet} (source: {doc}, p{page})")
        return "\n".join(lines) if lines else "(no matches)"

    # clamp 3–5 minutes (hard floor/ceiling)
    dur = max(PODCAST_MIN_MINUTES, min(PODCAST_MAX_MINUTES, float(duration_min or 3.0)))
    words_target = int(round(dur * TTS_WPM))

    style = (style or "podcast").strip().lower()
    speakers = 1 if style == "overview" else max(1, min(2, int(speakers)))

    # ~28 words ≈ ~160 chars; yields ~18–24 segments for 3–4 minutes
    segments_target = max(12, min(28, max(12, words_target // 28)))

    SYSTEM = (
        "You write *grounded* podcast scripts from PDF excerpts. "
        "Use ONLY the numbered context provided. Do NOT add outside facts or urls. "
        "Be concise, natural, and engaging."
    )
    USER = (
        f"Focus:\n{(focus or '').strip()}\n\n"
        f"Numbered context (must ground content/citations):\n{_ctx(ctx_items)}\n\n"
        "Write a podcast script as strict JSON with schema:\n"
        "{\n"
        '  "title": "short title",\n'
        '  "segments": [\n'
        '    {"speaker":"S1","text":"one utterance (plain text, no markdown)","refs":[1]},\n'
        '    {"speaker":"S2","text":"reply/contrast/example","refs":[2,3]},\n'
        "    ...\n"
        "  ]\n"
        "}\n"
        f"- Speakers: {'S1 only' if speakers == 1 else 'S1 and S2 alternating'}.\n"
        f"- Target total length: ~{words_target} words (MUST be ≥ {int(words_target*0.98)} and ≤ {int(words_target*1.15)}).\n"
        f"- Aim for ~{segments_target} segments.\n"
        f"- Each segment MUST be between {SEGMENT_CHARS_MIN} and {SEGMENT_CHARS_MAX} characters (no markdown).\n"
        "- Every segment must include refs pointing to relevant numbered context items.\n"
        "- Structure: open → deepen with contrasts/examples → wrap up with a takeaway.\n"
        "\n"
        "Also consider these Insights (bulleted markdown) as thematic hints; DO NOT invent citations from them:\n"
        f"{(insights_markdown or '—')}\n"
        "- Return ONLY the JSON object, no prose."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]

def _maybe_expand_plan(
    plan: Dict[str, Any],
    ctx_items: List[Dict[str, Any]],
    *,
    min_words: int,
    speakers: int,
) -> Dict[str, Any]:
    """If plan is short, expand in-place while keeping refs & alternation."""
    current = _plan_word_count(plan)
    if current >= min_words:
        return plan

    SYSTEM = (
        "You are revising a grounded podcast script to reach a longer target length.\n"
        "Use ONLY the provided numbered context. Keep title, speaker labels, and refs. Do NOT invent citations."
    )
    USER = (
        "Expand the following JSON script so that:\n"
        f"- Total words ≥ {min_words}\n"
        f"- Speakers: {'S1 only' if speakers == 1 else 'S1 and S2 alternating'}\n"
        f"- Each segment is between {SEGMENT_CHARS_MIN} and {SEGMENT_CHARS_MAX} characters (no markdown)\n"
        "- Maintain natural flow; keep refs pointing to the most relevant context items.\n\n"
        "Numbered context:\n" +
        "\n".join([
            f"[{i+1}] {(it.get('section_title') or '').strip()} — {(it.get('snippet') or it.get('content') or it.get('section_text') or '').strip()[:420]}"
            for i, it in enumerate(ctx_items)
        ]) +
        "\n\nExisting script JSON:\n" + json.dumps(plan, ensure_ascii=False)
    )
    try:
        out = get_llm_response([{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]) or ""
        expanded = _json_from_text(out) or plan
        return expanded
    except Exception:
        return plan

def _stitch_segments_to_mp3(segment_paths: List[Path], out_path: Path, gap_ms: int = 300) -> Path:
    """Concatenate mp3 segment files with small gaps.
    - Uses pydub+ffmpeg if available.
    - Otherwise, cleanly falls back to simple binary concat (still playable).
    - Silences pydub 'ffmpeg not found' warnings.
    """
    import shutil, warnings
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def simple_concat(paths: List[Path], dst: Path) -> Path:
        with open(dst, "wb") as fout:
            for fp in paths:
                with open(fp, "rb") as fin:
                    fout.write(fin.read())
        return dst

    # If ffmpeg isn't in PATH, skip pydub entirely
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return simple_concat(segment_paths, out_path)

    try:
        warnings.filterwarnings("ignore", module="pydub.utils")
        from pydub import AudioSegment
        silence = AudioSegment.silent(duration=max(0, int(gap_ms)))
        combined = None
        for p in segment_paths:
            seg = AudioSegment.from_file(str(p), format="mp3")
            combined = seg if combined is None else (combined + silence + seg)
        combined.export(str(out_path), format="mp3")
        return out_path
    except Exception:
        # Any pydub/ffmpeg hiccup -> fallback
        return simple_concat(segment_paths, out_path)

# --- Proper insights prompt builder (top-level) -------------------------------
def _build_insights_messages(user_focus: str, ctx_items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Build a compact, grounded prompt for the LLM to generate Insights with four sections:
    Key insights / "Did you know?" / Contradictions / Inspirations.
    Enforces a strict markdown skeleton and in-text bracketed citations [i].
    """
    def _context_block(items: List[Dict[str, Any]], max_snippet: int = 420) -> str:
        lines: List[str] = []
        for i, it in enumerate(items, 1):
            title = (it.get("section_title") or "").strip()
            snippet = (it.get("snippet") or it.get("content") or it.get("section_text") or "").strip()
            if len(snippet) > max_snippet:
                snippet = snippet[:max_snippet].rsplit(" ", 1)[0] + "…"
            doc = (it.get("document") or "?")
            page = int(it.get("page_number", 0)) + 1
            lines.append(f"[{i}] {title} — {snippet} (source: {doc}, p{page})")
        return "\n".join(lines) if lines else "(no matches)"

    CONTEXT = _context_block(ctx_items)

    SYSTEM = (
        "You generate grounded insights for a PDF reader. "
        "Use ONLY the numbered context provided; do not add outside facts. "
        "Be concise, helpful, and neutral."
    )

    FORMAT_RULES = (
        "Write markdown with EXACTLY these sections in this order:\n"
        "### Key insights\n"
        "- 3–5 bullets; ≤25 words each; each bullet MUST end with citations like [1], [2].\n"
        "### \"Did you know?\"\n"
        "- 1–2 short bullets; surprising but grounded; each with citations. If none, write '—'.\n"
        "### Contradictions / counterpoints\n"
        "- 1–3 bullets noting disagreements, caveats, or limitations; each with citations. If none, write '—'.\n"
        "### Inspirations & connections across docs\n"
        "- 1–3 bullets connecting ideas across different sources (e.g., [1] ↔ [3]); each with citations. If none, write '—'.\n"
        "Rules: No extra sections; no preamble or conclusion; never invent citations; keep to 160–220 total words."
    )

    USER = (
        f"Focus / selection:\n{(user_focus or '').strip()}\n\n"
        f"Numbered context (use for citations):\n{CONTEXT}\n\n"
        f"{FORMAT_RULES}"
    )

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ]

# ------------------------------------------------------------------------------
# Insights fallback for LLM failure
# ------------------------------------------------------------------------------
def _fallback_insights_markdown(user_focus: str, items: List[Dict[str, Any]]) -> str:
    """Deterministic fallback (no LLM). Uses top snippets to produce a minimal grounded markdown."""
    if not items:
        return ("### Key insights\n—\n\n"
                "### \"Did you know?\"\n—\n\n"
                "### Contradictions / counterpoints\n—\n\n"
                "### Inspirations & connections across docs\n—")

    def cite(i: int) -> str:
        return f"[{i}]"

    # Key insights: extract first sentence from up to 4 snippets
    bullets = []
    for i, it in enumerate(items[:4], 1):
        snip = (it.get("snippet") or it.get("section_text") or it.get("content") or "").strip()
        sent = re.split(r"(?<=[.!?])\s+", snip)[0][:130].rstrip(" .")
        if sent:
            bullets.append(f"- {sent}. {cite(i)}")
    if not bullets:
        bullets = ["- —"]

    # Did you know?: pick a concise fact line from the top item
    dyk = "—"
    if items:
        s0 = (items[0].get("snippet") or "").strip()
        if s0:
            dyk_sent = re.split(r"(?<=[.!?])\s+", s0)[-1][:110].rstrip(" .")
            if dyk_sent:
                dyk = f"- {dyk_sent}. {cite(1)}"

    # Contradictions: look for adversative cues across top snippets
    contra = "—"
    cues = ("however", "but ", "whereas", "contrary", "limitation", "caution")
    for i, it in enumerate(items[:5], 1):
        sn = (it.get("snippet") or "").lower()
        if any(c in sn for c in cues):
            contra = f"- Potential caveat in {it.get('section_title','this section')}. {cite(i)}"
            break

    # Inspirations: link the first two distinct docs if present
    insp = "—"
    if len(items) >= 2 and (items[0].get("document") != items[1].get("document")):
        insp = f"- Connect ideas in {cite(1)} and {cite(2)} for a broader view."

    md = [
        "### Key insights",
        *bullets,
        "",
        "### \"Did you know?\"",
        dyk,
        "",
        "### Contradictions / counterpoints",
        contra,
        "",
        "### Inspirations & connections across docs",
        insp,
    ]
    return "\n".join(md).strip()

@app.post("/insights")
def insights(body: InsightsRequest):
    t_start = _now()
    name = body.document
    page = int(body.page)
    top_k = int(body.top_k)
    selection = (body.selection or "").strip()

    query_full = selection or _query_from_doc_page(name, page)
    query = _prepare_query_for_embedding(query_full)

    raw = _search_raw_cached(query, topn=max(20, top_k))
    used = _normalize_related_items(raw, query=query, min_score_pct=MIN_SCORE_PCT, top_k=top_k)

    # If nothing survives thresholding, still try with the best few so the UI isn't empty
    if not used:
        used = _normalize_related_items(raw, query=query, min_score_pct=0.0, top_k=min(3, max(1, top_k)))

    messages = _build_insights_messages(selection or query, used)

    try:
        text = (get_llm_response(messages) or "").strip()
        # If the model ignored format or returned empty, fall back
        if not text or not text.lstrip().startswith("### Key insights"):
            text = _fallback_insights_markdown(selection or query, used)
        mode = "structured_markdown"
    except Exception:
        # Never 500 here; surface a grounded fallback instead for a smoother UX
        text = _fallback_insights_markdown(selection or query, used)
        mode = "fallback"

    timings = {
        "total_ms": int((_now() - t_start) * 1000),
        "cache_entries": len(SEARCH_CACHE),
        "epoch": INDEX_EPOCH,
    }
    return {
        "text": text,
        "used_items": used,
        "provider": os.getenv("LLM_PROVIDER", "gemini"),
        "mode": mode,
        "timings": timings,
    }

@app.post("/podcast")
def podcast(body: Dict[str, Any]):
    """
    Generate audio:
    - If text is provided: synthesize directly (backwards compatible).
    - Else: build a grounded podcast/overview from selection + related docs + insights, then synthesize.
    Request (either of):
      { "text": "...", "voice": "alloy", "format": "audio-48khz-192kbitrate-mono-mp3" }
    or
      {
        "document": "foo.pdf", "page": 3, "selection": "text...",
        "style": "podcast" | "overview", "speakers": 2, "duration_min": 3.0,
        "voices": ["alloy","verse"],
        "format": "audio-48khz-192kbitrate-mono-mp3"
      }
    """
    # Back-compat raw text path
    text = (body.get("text") or "").strip()
    if text:
        prov = (os.getenv("TTS_PROVIDER", "azure") or "").lower()
        default_voice = (
            os.getenv("AZURE_TTS_VOICE", "alloy") if prov == "azure"
            else os.getenv("GCP_TTS_VOICE", "en-US-Neural2-F")
        )
        voice = _coerce_voice_for_provider(body.get("voice")) or default_voice
        fmt = body.get("format", "audio-48khz-192kbitrate-mono-mp3")
        path = synthesize(text, voice=voice, fmt=fmt)
        if (not path.exists()) or path.stat().st_size == 0:
            return JSONResponse(
                {
                    "error": "TTS synthesis produced no audio",
                    "provider": os.getenv("TTS_PROVIDER", ""),
                    "voice_used": voice,
                },
                status_code=502,
            )
        rel = f"/data/audio/{path.name}"
        return {"audio_url": rel, "bytes": path.stat().st_size, "mode": "raw"}

    # Structured podcast path
    try:
        req = PodcastGenRequest(**body)
    except Exception as e:
        return JSONResponse({"error": f"bad request: {e}"}, status_code=400)

    focus = (req.selection or "").strip()
    if not focus and req.document:
        try:
            focus = _query_from_doc_page(req.document, int(req.page or 0))
        except Exception:
            pass
    if not focus:
        return JSONResponse({"error": "Provide either 'text' or ('selection' and document/page)."}, status_code=400)

    # Clamp duration and compute targets
    eff_minutes = max(PODCAST_MIN_MINUTES, min(PODCAST_MAX_MINUTES, float(req.duration_min or 3.0)))
    min_words_needed = int(round(eff_minutes * TTS_WPM * EXPANSION_TOLERANCE))

    # Pull related items (reuse the same quality gating as recommendations/insights)
    top_k = 5 if req.speakers != 1 else 3
    query = _prepare_query_for_embedding(focus)
    raw = _search_raw_cached(query, topn=max(20, top_k))
    used = _normalize_related_items(raw, query=query, min_score_pct=MIN_SCORE_PCT, top_k=top_k)
    if not used:
        used = _normalize_related_items(raw, query=query, min_score_pct=0.0, top_k=min(3, top_k))

    # Build Insights markdown (bulb) from the same context and feed to the podcast plan
    insights_md = ""
    try:
        insights_msgs = _build_insights_messages(focus, used)
        insights_md = get_llm_response(insights_msgs) or ""
    except Exception:
        insights_md = ""

    # Ask LLM for a podcast plan (now includes insights as hints)
    messages = _build_podcast_messages(
        focus,
        used,
        style=req.style,
        speakers=req.speakers,
        duration_min=eff_minutes,
        insights_markdown=insights_md,
    )
    try:
        plan_text = get_llm_response(messages) or ""
    except Exception:
        plan_text = ""

    plan = _json_from_text(plan_text) or {"title": "Audio overview", "segments": [{"speaker": "S1", "text": focus, "refs": []}]}
    segments = plan.get("segments") or []
    if req.style.lower() == "overview":
        # force single speaker
        for seg in segments:
            seg["speaker"] = "S1"

    # If short, expand up to twice
    tries = 0
    while _plan_word_count(plan) < min_words_needed and tries < 2:
        plan = _maybe_expand_plan(plan, used, min_words=min_words_needed, speakers=(1 if req.style.lower() == "overview" else 2))
        segments = plan.get("segments") or []
        tries += 1

    # Synthesize per segment (alternate voices) and stitch
    voices = _pick_voices(max(1, req.speakers), req.voices)
    fmt = req.format or "audio-48khz-192kbitrate-mono-mp3"

    seg_paths: List[Path] = []
    for i, seg in enumerate(segments):
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        spk = (seg.get("speaker") or ("S1" if i % 2 == 0 else "S2")).upper()
        vidx = 0 if spk == "S1" else 1
        voice = voices[min(vidx, len(voices)-1)]
        voice = _coerce_voice_for_provider(voice)  # ensure valid for Azure OpenAI TTS
        try:
            p = synthesize(txt, voice=voice, fmt=fmt)
            if p.exists() and p.stat().st_size > 0:
                seg_paths.append(p)
            else:
                print(f"[WARN] Empty TTS output for segment {i+1} using voice '{voice}'")
        except Exception as e:
            print("[WARN] TTS failed for a segment:", e)

    if not seg_paths:
        return JSONResponse({"error": "Failed to synthesize audio"}, status_code=500)

    out_name = f"podcast_{uuid.uuid4().hex}.mp3"
    out_path = AUDIO_DIR / out_name
    final_path = _stitch_segments_to_mp3(seg_paths, out_path, gap_ms=300)
    rel = f"/data/audio/{final_path.name}"

    est_secs = int(round((_plan_word_count(plan) / max(1, TTS_WPM)) * 60))

    # Optionally cleanup temp segment files (they live in data/audio already; keep for debugging if desired)
    # for p in seg_paths:
    #     try: p.unlink(missing_ok=True)
    #     except Exception: pass

    return {
        "audio_url": rel,
        "title": plan.get("title", "Podcast"),
        "script": segments,
        "used_items": used,
        "provider": os.getenv("TTS_PROVIDER", "azure"),
        "mode": req.style.lower(),
        "estimated_duration_sec": est_secs
    }

@app.post("/select")
def select_flow(body: SelectRequest):
    t_start = _now()
    sel = (body.selection or "").strip()
    if not sel:
        return JSONResponse({"error": "selection is required"}, status_code=400)
    top_k = int(body.top_k)

    query = _prepare_query_for_embedding(sel)
    raw = _search_raw_cached(query, topn=max(20, top_k))
    hits = _normalize_related_items(raw, query=query, min_score_pct=MIN_SCORE_PCT, top_k=top_k)

    elapsed_ms = int((_now() - t_start) * 1000)
    budget_left_s = max(0.0, (RECO_BUDGET_MS - elapsed_ms) / 1000.0)
    hits = _maybe_llm_rerank(query, hits, top_k=top_k, time_budget_left=budget_left_s)

    messages = _build_insights_messages(sel, hits)

    try:
        insight_text = get_llm_response(messages)
        insight_text = (insight_text or '').strip()
        timings = {
            "total_ms": int((_now() - t_start) * 1000),
            "cache_entries": len(SEARCH_CACHE),
            "epoch": INDEX_EPOCH,
        }
        return {"results": hits, "insight": insight_text, "provider": os.getenv("LLM_PROVIDER", "gemini"), "timings": timings}
    except Exception as e:
        return JSONResponse(
            {
                "error": "LLM call failed",
                "detail": repr(e),
                "results": hits,
                "provider": os.getenv("LLM_PROVIDER", "gemini"),
                "timings": {"total_ms": int((_now() - t_start) * 1000)},
            },
            status_code=500,
        )

# ==============================================================================
# Notes
# - Swagger UI is now at /swagger (so GET /docs lists your PDFs).
# - LLM calls go straight through chat_with_llm.get_llm_response(messages).
# - Works with Gemini (GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS),
#   Azure OpenAI, OpenAI, or Ollama based on LLM_PROVIDER.
# ==============================================================================