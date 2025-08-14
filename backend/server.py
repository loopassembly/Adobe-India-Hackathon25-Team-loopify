#!/usr/bin/env python3
# FastAPI backend optimized for ≤10s end-to-end with LLM/TTS optional paths.

import os, time, re, sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pdfplumber
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from process_pdfs import process_file  # local extractor

# ---------------- small text utils ----------------
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z\-\’']+", (s or "").lower())

def looks_like_heading(ln: str) -> bool:
    if not ln or not ln.strip(): return False
    s = ln.strip()
    if s.endswith(".") or len(s) > 90: return False
    if re.search(r"\b(this guide|you'll|we'll|we will|help you|to help you|this section)\b", s, re.I): return False
    if sum(ch in s for ch in ",;:") > 2: return False
    words = re.findall(r"[A-Za-z][A-Za-z’'\-]*", s)
    if len(words) < 2 or len(words) > 14: return False
    if re.match(r"^(ultimate|comprehensive)\s+guide\b", s, re.I): return True
    if re.match(r"^(guide to|things to do|restaurants|hotels|tips|history|culture|traditions|itinerary|day \d+|cities)\b", s, re.I):
        return True
    title_case_ratio = sum(w[0].isupper() for w in words) / len(words)
    return title_case_ratio >= 0.7 or s.isupper() or bool(re.compile(r"^[A-Z][\w’'&\- ]+(?:\s*\([^)]*\))?$").match(s))

def _clean_bullets(s: str) -> str:
    return re.sub(r"\s{2,}", " ", (s or "").replace("•", "- ")).strip()

# --------------- domain detection + rule boosts ----------------
def detect_domain(persona: str, job: str) -> str:
    t = f"{persona} {job}".lower()
    if any(w in t for w in ["trip","travel","itinerary","vacation","tour","planner"]): return "travel"
    if any(w in t for w in ["menu","buffet","dinner","vegetarian","gluten","cater","food"]): return "food"
    if any(w in t for w in ["acrobat","pdf","e-signature","convert","export","edit","share","form","forms","ai"]): return "acrobat"
    return "generic"

MEAT = {"beef","pork","chicken","turkey","lamb","mutton","veal","fish","salmon","tuna","shrimp","prawn","anchovy","bacon","ham","sausage","prosciutto","pancetta","duck"}
GLUTEN = {"bread","baguette","bun","pita","naan","roti","tortilla","wrap","quesadilla","pasta","spaghetti","noodles","lasagna","lasagne","ziti","macaroni","pretzel","flour","batter","breadcrumbs","croutons","gnocchi"}
VEG  = {"veg","vegetarian","paneer","dal","lentil","chickpea","hummus","falafel","tofu","ratatouille","salad","rice","quinoa","polenta","corn","potato","vegetable","stir-fry","sushi","rolls","baba","ganoush","tahini"}
BUFFET = {"salad","rice","pulao","pilaf","dal","hummus","baba ganoush","ratatouille","falafel","paneer","polenta","casserole","bake","tray","platter","dip","sushi","rolls"}

TRAVEL_KW = {"things to do","attractions","must-see","must see","highlights","restaurants","where to eat","hotels","where to stay","tips and tricks","getting around","transport","history","culture","traditions","day trips","beaches","itinerary","day 1","day 2","day 3","day 4","budget","museums","guide","cities"}
ATTRACTION_PAT = re.compile(r"^[A-Z][\w’'&\- ]+(?:\s*\([^)]*\))?$")

FORM_HEADS = {"prepare form","prepare forms","fillable","flat form","interactive form","form field","change flat forms to fillable","e-signature","e-signatures","signature","signatures","fill and sign","send for signature","request e-signatures","request signatures"}
SHARE_HEADS = {"share","get link","copy link","review","comment","comments","mark up","markup","send link","unshare","add people","whatsapp","teams","gmail","outlook"}
CONVERT_HEADS  = {"convert","export","word","ppt","powerpoint","excel","image","jpg","jpeg","png","html","xml","clipboard","multiple pdfs","multiple files","portfolio"}

def rule_boost(title: str, doc_name: str, context: str, persona: str, job: str, domain: str) -> float:
    tkn = set(tokenize(title) + tokenize(context))
    score = 0.0
    if domain == "food":
        if tkn & {m.lower() for m in MEAT}: score -= 0.5
        if "chicken" in tkn and "broth" in tkn: score -= 0.45
        score += 0.12 * len(tkn & {v.lower() for v in VEG})
        if tkn & {g.lower() for g in GLUTEN}: score -= 0.12
        score += 0.10 * len({"rice","quinoa","polenta","corn","potato","dal","lentil","chickpea"} & tkn)
        dn = (doc_name or "").lower()
        if any(w in dn for w in ("dinner","mains","sides")): score += 0.14
        score += 0.10 * len(tkn & {b.lower() for b in BUFFET})
        if "sides" in dn: score += 0.26
        if "breakfast" in dn: score -= 0.5
        if "lunch" in dn: score -= 0.22

    elif domain == "travel":
        lower_title = (title or "").lower()
        score += 0.14 * sum(1 for kw in TRAVEL_KW if kw in lower_title)
        if re.match(r"^(ultimate|comprehensive)\s+guide\b", (title or ""), re.I): score += 0.20
        if re.match(r"^\s*things to do\b", (title or ""), re.I): score += 0.24
        if ATTRACTION_PAT.match(title or "") and len(title.split()) <= 12: score += 0.12
        dn = (doc_name or "").lower()
        if "things to do" in dn: score += 0.45
        if any(w in dn for w in ["restaurants","hotels"]): score += 0.10
        if "cities" in dn: score += 0.08
        if any(w in dn for w in ["traditions","culture"]): score += 0.06
        if "tips and tricks" in dn or "tips" in dn: score -= 0.22
        if "packing" in lower_title: score -= 0.20

    elif domain == "acrobat":
        lt = (title or "").lower()
        form_hits = sum(1 for kw in FORM_HEADS if kw in lt); score += 0.30 * form_hits
        share_hits = sum(1 for kw in SHARE_HEADS if kw in lt); score += 0.14 * share_hits
        conv_hits  = sum(1 for kw in CONVERT_HEADS if kw in lt); score += 0.14 * conv_hits
        if "change flat forms to fillable" in lt: score += 0.40
        if "fill and sign" in lt: score += 0.26
        if any(k in lt for k in ["request e-signatures","send for signature","send a document to get signatures"]): score += 0.26
        if "prepare form" in lt or "prepare forms" in lt: score += 0.22
        if "leader in e-signatures" in lt: score -= 0.35
        if lt.startswith(("consider ","note:","what if","tip:","warning:","faq")) or "?" in lt: score -= 0.22

    pj = set(tokenize(persona) + tokenize(job))
    score += 0.03 * len(pj & tkn)
    return score

MEAT_TOKENS = {t.lower() for t in MEAT}
GLUTEN_TOKENS = {t.lower() for t in GLUTEN}

def strip_meat_lines(s: str) -> str:
    out = []
    for ln in (s or "").splitlines():
        toks = set(tokenize(ln))
        if toks & MEAT_TOKENS:
            continue
        out.append(ln)
    return _clean_bullets("\n".join(out))

def looks_gluten_free(s: str) -> bool:
    toks = set(tokenize(s))
    return not (toks & GLUTEN_TOKENS)

# --------------- snippet extraction anchored to the heading ---------------
def refine_text_from_cache(pages_text: List[str], page_no: int, domain: str, title: str | None = None) -> str:
    text = pages_text[page_no] if 0 <= page_no < len(pages_text) else ""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    if title:
        title_lc = title.strip().lower()
        idx = next((i for i, ln in enumerate(lines) if ln.lower() == title_lc), None)
        if idx is None:
            idx = next((i for i, ln in enumerate(lines) if title_lc in ln.lower()), None)
        if idx is not None:
            chunk = [lines[idx]]
            i = idx + 1
            while i < len(lines) and len(" ".join(chunk)) < 700:
                if looks_like_heading(lines[i]) and i > idx + 1:
                    break
                chunk.append(lines[i]); i += 1
            return _clean_bullets(" ".join(chunk[:8]))
    sents = [s.strip() for s in SENT_SPLIT.split(text or "") if s.strip()]
    return _clean_bullets(" ".join(sents[:3]))

# ---------------- app + storage ----------------
BASE = Path(__file__).parent
UPLOAD_DIR = BASE / "data" / "uploads"
AUDIO_DIR  = BASE / "data" / "audio"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="PDF Intelligence Backend (≤10s)",
    docs_url="/api-docs",
    redoc_url=None
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.mount("/data/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/data/audio",   StaticFiles(directory=AUDIO_DIR),   name="audio")
# Alias so the frontend default (/data/pdfs) works without extra config
app.mount("/data/pdfs", StaticFiles(directory=UPLOAD_DIR), name="pdfs")

DOCS: Dict[str, Dict[str, Any]] = {}   # name -> {path, pages_text, outline}
CHUNKS: List[Dict[str, Any]] = []      # {doc, page, title, text}
VEC: TfidfVectorizer | None = None
MATRIX = None
DOMAIN = "generic"
PERSONA = ""
JOB = ""

RECS_CACHE: Dict[Tuple, Tuple[float, Any]] = {}
INSIGHTS_CACHE: Dict[Tuple, Tuple[float, Any]] = {}
TTS_CACHE: Dict[str, str] = {}

def get_pages_text(pdf_path: Path) -> List[str]:
    texts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for p in pdf.pages:
            texts.append(p.extract_text() or "")
    return texts

def level_rank(h: str) -> int:
    m = re.search(r'(\d+)', h or "")
    return int(m.group(1)) if m else 7

def build_chunks(doc_name: str, pdf_path: Path) -> List[Dict[str, Any]]:
    # VERY IMPORTANT: never let a PDF crash indexing
    try:
        out = process_file(pdf_path)
    except Exception as e:
        print(f"[WARN] process_file failed on {pdf_path.name}: {e}", file=sys.stderr)
        out = {"outline": []}

    outline = out.get("outline", []) or []
    per_page: Dict[int, List[Dict[str, Any]]] = {}
    for h in outline:
        p = int(h.get("page", 0))
        per_page.setdefault(p, []).append(h)

    chunks = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for pno, page in enumerate(pdf.pages):
            heading = None
            if pno in per_page and per_page[pno]:
                # choose strongest (lowest H number)
                heading = sorted(per_page[pno], key=lambda x: level_rank(x.get("level","H7")))[0].get("text")
            else:
                lines = [ln.strip() for ln in (page.extract_text() or "").splitlines()]
                for ln in lines:
                    if looks_like_heading(ln):
                        heading = ln.strip(); break
            if not heading: heading = f"Page {pno+1}"
            text = page.extract_text() or ""
            chunks.append({"doc": doc_name, "page": pno, "title": heading, "text": text})
    return chunks

def rebuild_index():
    global VEC, MATRIX
    corpus = [(c["title"] + " " + (c["text"] or "")).strip() for c in CHUNKS]
    VEC = TfidfVectorizer(max_features=18000, ngram_range=(1,2))
    MATRIX = VEC.fit_transform(corpus)

# ---------------- models ----------------
class IndexResponse(BaseModel):
    status: str
    domain: str
    docs_added: List[str]
    num_chunks: int

class RecReq(BaseModel):
    document: str
    page: int | None = None
    title: str | None = None
    top_k: int = 5

class InsightReq(BaseModel):
    document: str
    page: int
    top_k: int = 3

class PodcastAutoReq(BaseModel):
    document: str
    page: int
    minutes: int = 3   # 2..5
    voice: str | None = "en-US-JennyNeural"
    format: str | None = "audio-48khz-192kbitrate-mono-mp3"

# ADD THIS MODEL (it was missing)
class PodcastReq(BaseModel):
    text: str
    voice: str | None = "en-US-JennyNeural"
    format: str | None = "audio-48khz-192kbitrate-mono-mp3"

# ---------------- routes ----------------
@app.get("/health")
def health():
    return {"ok": True, "docs": list(DOCS.keys()), "chunks": len(CHUNKS), "domain": DOMAIN}


@app.middleware("http")
async def add_static_headers(request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith("/data/pdfs/"):
        resp.headers["Accept-Ranges"] = "bytes"
        resp.headers["Access-Control-Expose-Headers"] = "Accept-Ranges, Content-Length, Content-Range"
    return resp



@app.post("/index", response_model=IndexResponse)
async def index_docs(
    persona: str = Form(...),
    job_to_be_done: str = Form(...),
    files: List[UploadFile] = File(...)
):
    global DOCS, CHUNKS, DOMAIN, PERSONA, JOB, RECS_CACHE, INSIGHTS_CACHE
    PERSONA, JOB = persona, job_to_be_done
    DOMAIN = detect_domain(PERSONA, JOB)
    RECS_CACHE.clear(); INSIGHTS_CACHE.clear()

    added = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue
        dest = UPLOAD_DIR / f.filename
        dest.write_bytes(await f.read())
        pages_text = get_pages_text(dest)
        DOCS[f.filename] = {"path": dest, "pages_text": pages_text, "outline": None}
        CHUNKS.extend(build_chunks(f.filename, dest))
        added.append(f.filename)

    if not added:
        raise HTTPException(400, "No PDFs uploaded")

    rebuild_index()
    return IndexResponse(status="ok", domain=DOMAIN, docs_added=added, num_chunks=len(CHUNKS))

@app.get("/docs")
def list_docs():
    out = [{"name": name, "pages": len(meta.get("pages_text", []))} for name, meta in DOCS.items()]
    return {"documents": out, "domain": DOMAIN, "persona": PERSONA, "job": JOB}

@app.get("/outline")
def get_outline(document: str = Query(...)):
    if document not in DOCS:
        raise HTTPException(404, "Unknown document")
    if DOCS[document].get("outline") is None:
        DOCS[document]["outline"] = process_file(DOCS[document]["path"])
    res = DOCS[document]["outline"]
    return {"title": res.get("title",""), "outline": res.get("outline", [])}

@app.post("/recommendations")
def recommendations(req: RecReq):
    if req.document not in DOCS:
        raise HTTPException(404, "Unknown document")
    if VEC is None or MATRIX is None or not CHUNKS:
        raise HTTPException(400, "Index empty. Upload via /index first.")

    key = (req.document, req.page, req.title, DOMAIN, PERSONA, JOB, req.top_k)
    cached = RECS_CACHE.get(key)
    now = time.time()
    if cached and now - cached[0] < 120:
        return cached[1]

    q_title = req.title
    if q_title is None and req.page is not None:
        try:
            q_title = next(c["title"] for c in CHUNKS if c["doc"] == req.document and c["page"] == req.page)
        except StopIteration:
            q_title = f"Page {req.page+1}"
    if not q_title:
        q_title = "current section"

    q_text = ""
    if req.page is not None and 0 <= req.page < len(DOCS[req.document]["pages_text"]):
        sents = [s.strip() for s in SENT_SPLIT.split(DOCS[req.document]["pages_text"][req.page] or "") if s.strip()]
        q_text = " ".join(sents[:3])
    q = (q_title + " " + q_text).strip()

    qv = VEC.transform([q])
    sims = cosine_similarity(MATRIX, qv).ravel()
    smin, smax = float(sims.min()), float(sims.max())
    sims_scaled = (sims - smin) / (smax - smin + 1e-9)
    boosts = np.array([
        rule_boost(c["title"], c["doc"], " ".join((c["text"] or "").split()[:60]), PERSONA, JOB, DOMAIN)
        for c in CHUNKS
    ])
    final = 0.6 * sims_scaled + boosts
    order = np.argsort(-final)

    per_doc_cap = 1 if DOMAIN == "food" else 2
    seen: Dict[str, int] = {}
    results = []
    for idx in order:
        c = CHUNKS[idx]
        if c["doc"] == req.document and req.page is not None and c["page"] == req.page:
            continue
        if seen.get(c["doc"], 0) >= per_doc_cap:
            continue

        raw_snip = refine_text_from_cache(
            DOCS[c["doc"]]["pages_text"], c["page"], DOMAIN, title=c["title"]
        )
        snip = strip_meat_lines(raw_snip) if DOMAIN == "food" else raw_snip

        results.append({
            "document": c["doc"],
            "section_title": c["title"],
            "page_number": c["page"],
            "score": float(final[idx]),
            "snippet": snip,
            "gluten_free_guess": looks_gluten_free(snip) if DOMAIN == "food" else None
        })
        seen[c["doc"]] = seen.get(c["doc"], 0) + 1
        if len(results) == max(1, req.top_k):
            break

    payload = {"items": results, "domain": DOMAIN, "query_used": q_title}
    RECS_CACHE[key] = (now, payload)
    return payload

# -------------------- LLM Insights (timeout + cache) --------------------
async def _call_llm_async(prompt: str, max_tokens=200, timeout=4.5) -> str:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if provider == "azure":
                base = os.environ["AZURE_OPENAI_BASE"].rstrip("/")
                key  = os.environ["AZURE_OPENAI_KEY"]
                api_version = os.environ.get("AZURE_API_VERSION", "2024-02-01")
                deployment = os.environ["AZURE_DEPLOYMENT_NAME"]
                url = f"{base}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
                headers = {"api-key": key, "Content-Type": "application/json"}
                body = {"messages":[{"role":"system","content":"Be concise and factual."},{"role":"user","content":prompt}],
                        "temperature":0.2,"max_tokens":max_tokens}
                r = await client.post(url, headers=headers, json=body)
                r.raise_for_status()
                js = r.json()
                return js["choices"][0]["message"]["content"].strip()
            else:
                api_key = os.environ["OPENAI_API_KEY"]
                base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
                model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                url = f"{base}/chat/completions"
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
                body = {"model":model,
                        "messages":[{"role":"system","content":"Be concise and factual."},{"role":"user","content":prompt}],
                        "temperature":0.2,"max_tokens":max_tokens}
                r = await client.post(url, headers=headers, json=body)
                r.raise_for_status()
                js = r.json()
                return js["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(LLM timeout/fallback: {e})"

@app.post("/insights")
async def insights(req: InsightReq):
    if req.document not in DOCS:
        raise HTTPException(404, "Unknown document")

    key = (req.document, req.page, DOMAIN, PERSONA, JOB, req.top_k)
    cached = INSIGHTS_CACHE.get(key)
    now = time.time()
    if cached and now - cached[0] < 3600:
        return cached[1]

    cur_text = ""
    if 0 <= req.page < len(DOCS[req.document]["pages_text"]):
        cur_text = DOCS[req.document]["pages_text"][req.page]
    if DOMAIN == "food":
        cur_text = strip_meat_lines(cur_text)

    recs = recommendations(RecReq(document=req.document, page=req.page, title=None, top_k=req.top_k))
    pieces = [f"- {it['section_title']} ({it['document']} p{it['page_number']+1}): {it['snippet']}" for it in recs["items"]]
    context = "\n".join(pieces[:req.top_k])

    prompt = f"""You summarize PDF sections for a vegetarian dinner menu (gluten-free preferred).
- Do NOT mention meat/seafood/poultry.
- Be concise and factual.

Current section (trimmed to 900 chars):
---
{cur_text[:900]}
---

Top related snippets:
{context}

In 2–3 short sentences: key insight + one 'did you know?' fact (vegetarian/gluten-free)."""
    text = await _call_llm_async(prompt, max_tokens=180, timeout=4.5)

    if text.startswith("(LLM timeout") or len(text.split()) < 5:
        sents = [s.strip() for s in SENT_SPLIT.split(cur_text) if s.strip()]
        prime = " ".join(sents[:2])[:300]
        didyou = recs["items"][0]["snippet"] if recs["items"] else ""
        text = _clean_bullets(f"{prime} Did you know? {didyou}")

    payload = {"insight": text, "used_items": recs["items"]}
    INSIGHTS_CACHE[key] = (now, payload)
    return payload

# -------------------- Azure TTS --------------------

# -------------------- Long-form Podcast Generation (Auto) --------------------
@app.post("/podcast_auto")
async def podcast_auto(req: PodcastAutoReq):
    """
    Build a ~2–5 minute narrated audio from the current page + related snippets.
    Expands to a natural script with the LLM, then synthesizes via Azure TTS.
    """
    if req.document not in DOCS:
        raise HTTPException(404, "Unknown document")

    # Only Azure TTS supported here (to match evaluation env)
    if os.getenv("TTS_PROVIDER", "azure").lower() != "azure":
        raise HTTPException(400, "Only Azure TTS is supported here.")
    key = os.getenv("AZURE_TTS_KEY")
    endpoint = os.getenv("AZURE_TTS_ENDPOINT")
    if not key or not endpoint:
        raise HTTPException(400, "AZURE_TTS_KEY/AZURE_TTS_ENDPOINT missing.")

    # Clamp minutes to 2..5 and target ~160 wpm
    minutes = max(2, min(5, int(req.minutes or 3)))
    target_words = 160 * minutes

    # Pull current page text and top related snippets
    pages = DOCS[req.document].get("pages_text", [])
    cur_text = pages[req.page] if 0 <= req.page < len(pages) else ""

    recs_payload = recommendations(RecReq(document=req.document, page=req.page, title=None, top_k=6))
    bullets = "\n".join(
        f"- {it['section_title']} (p{it['page_number']+1} in {it['document']}): {it['snippet']}"
        for it in recs_payload.get("items", [])
    )

    # Ask the LLM to shape a long, listener-friendly script
    prompt = f"""Write a podcast script (second-person narrator) about this PDF section.
Target length ≈ {target_words} words (±10%). Use plain English, short paragraphs.
Include: quick intro, key ideas from the current section, 3–6 related snippets as context, and a wrap-up.

Current section text:
---
{(cur_text or '')[:1800]}
---

Related snippets:
{bullets}
"""
    script = await _call_llm_async(prompt, max_tokens=1200, timeout=6.0)

    # If LLM under-shoots, pad with trimmed source content
    if len(script.split()) < target_words * 0.5:
        base = (cur_text or "") + "\n\n" + bullets
        while len(script.split()) < target_words * 0.8 and base:
            script += "\n\n" + base[:1000]
            base = base[1000:]

    # SSML pacing: slightly slower for longer durations
    rate = "slow" if minutes >= 4 else "medium"
    voice = req.voice or "en-US-JennyNeural"
    out_format = req.format or "audio-48khz-192kbitrate-mono-mp3"
    ssml = f"<speak version='1.0' xml:lang='en-US'><voice name='{voice}'><prosody rate='{rate}'>{script}</prosody></voice></speak>"

    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": out_format,
        "User-Agent": "pdf-intel-backend"
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(endpoint, headers=headers, content=ssml.encode("utf-8"))
            r.raise_for_status()
            # Persist audio
            h = str(abs(hash((req.document, req.page, minutes, script[:64]))))
            fname = f"tts_{int(time.time())}_{h}.mp3"
            (AUDIO_DIR / fname).write_bytes(r.content)
            return {"audio_url": f"/data/audio/{fname}", "approx_minutes": minutes, "words": len(script.split())}
    except Exception as e:
        raise HTTPException(504, f"TTS timeout/error: {e}")


@app.post("/podcast")
async def podcast(req: PodcastReq):
    if os.getenv("TTS_PROVIDER","azure").lower() != "azure":
        raise HTTPException(400, "Only Azure TTS is supported here.")
    key = os.getenv("AZURE_TTS_KEY")
    endpoint = os.getenv("AZURE_TTS_ENDPOINT")
    if not key or not endpoint:
        raise HTTPException(400, "AZURE_TTS_KEY/AZURE_TTS_ENDPOINT missing.")

    text = (req.text or "").strip()
    if len(text) > 700:
        text = text[:700] + "…"

    h = str(abs(hash((text, req.voice, req.format))))
    if h in TTS_CACHE:
        fname = TTS_CACHE[h]
        return {"audio_url": f"/data/audio/{fname}", "cached": True}

    ssml = f"<speak version='1.0' xml:lang='en-US'><voice name='{req.voice}'>{text}</voice></speak>"

    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": req.format,
        "User-Agent": "pdf-intel-backend"
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(endpoint, headers=headers, content=ssml.encode("utf-8"))
            r.raise_for_status()
            fname = f"tts_{int(time.time())}_{h}.mp3"
            (AUDIO_DIR / fname).write_bytes(r.content)
            TTS_CACHE[h] = fname
            return {"audio_url": f"/data/audio/{fname}", "bytes": len(r.content)}
    except Exception as e:
        raise HTTPException(504, f"TTS timeout/error: {e}")

@app.get("/")
def root():
    return PlainTextResponse("PDF Intelligence Backend running. See /docs")
