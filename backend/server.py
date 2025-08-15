# server.py
import os
import asyncio
import importlib
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from process_pdfs import process_file
import retrieval  # we may reload this module when switching model names
from audio_bridge import synthesize
from chat_with_llm import get_llm_response  # <-- use the provided file directly

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
app = FastAPI(title="DocDots Backend", version="3.5", docs_url="/swagger", redoc_url=None)

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
        "llm_provider": STATUS["provider"],
        "data_dir": str(DATA_DIR),
        "swagger": "/swagger",
    }

@app.get("/status")
def status():
    return {
        "phase": STATUS["phase"],
        "progress": STATUS["progress"],
        "message": STATUS["message"],
        "embedding_model": STATUS["model"],
    }

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
    name = body.document
    page = int(body.page)
    top_k = int(body.top_k)
    selection = (body.selection or "").strip()
    query = selection or _query_from_doc_page(name, page)
    results = retrieval.search(query, top_k=top_k)
    return {"results": results, "query_used": query}

def _build_insights_messages(user_focus: str, ctx_items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    # system guidance (keep concise; your provided file uses Chat models expecting messages)
    SYSTEM = (
        "You are generating grounded insights for a PDF reading app. "
        "Only use the provided context; avoid outside knowledge. "
        "Write one compact paragraph of key takeaways. Then add one short line that starts with "
        "\"Did you know?\" if appropriate. Keep under 120 words total."
    )
    # build a compact context block
    lines = []
    for i, it in enumerate(ctx_items, 1):
        lines.append(
            f"[{i}] {it['document']} p{it['page_number']+1}: "
            f"{it['section_title']} — {it.get('snippet','')}"
        )
    CONTEXT = "\n".join(lines) if lines else "(no matches)"
    USER = f"User focus: {user_focus}\n\nContext:\n{CONTEXT}"
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ]

@app.post("/insights")
def insights(body: InsightsRequest):
    name = body.document
    page = int(body.page)
    top_k = int(body.top_k)
    selection = (body.selection or "").strip()

    query = selection or _query_from_doc_page(name, page)
    used = retrieval.search(query, top_k=top_k)

    messages = _build_insights_messages(selection or query, used)

    try:
        text = get_llm_response(messages)
        return {"text": (text or "").strip(), "used_items": used, "provider": os.getenv("LLM_PROVIDER", "gemini"), "mode": "direct"}
    except Exception as e:
        return JSONResponse(
            {
                "error": "LLM call failed",
                "detail": repr(e),
                "used_items": used,
                "provider": os.getenv("LLM_PROVIDER", "gemini"),
                "mode": "direct",
            },
            status_code=500,
        )

@app.post("/podcast")
def podcast(body: Dict[str, Any]):
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)
    voice = body.get("voice", "en-US-JennyNeural")
    fmt = body.get("format", "audio-48khz-192kbitrate-mono-mp3")
    path = synthesize(text, voice=voice, fmt=fmt)
    rel = f"/data/audio/{path.name}"
    return {"audio_url": rel, "bytes": path.stat().st_size}

@app.post("/select")
def select_flow(body: SelectRequest):
    sel = (body.selection or "").strip()
    if not sel:
        return JSONResponse({"error": "selection is required"}, status_code=400)
    top_k = int(body.top_k)

    hits = retrieval.search(sel, top_k=top_k)
    messages = _build_insights_messages(sel, hits)

    try:
        insight_text = get_llm_response(messages)
        return {"results": hits, "insight": (insight_text or '').strip(), "provider": os.getenv("LLM_PROVIDER", "gemini")}
    except Exception as e:
        return JSONResponse(
            {
                "error": "LLM call failed",
                "detail": repr(e),
                "results": hits,
                "provider": os.getenv("LLM_PROVIDER", "gemini"),
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