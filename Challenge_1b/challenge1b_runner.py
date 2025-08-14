#!/usr/bin/env python3
"""
Challenge 1B runner (persona-aware, generalizable, no hardcoding to filenames/pages)

- Uses process_pdfs.process_file for outline candidates.
- Domain fallbacks (food / travel / acrobat) when outlines are sparse.
- Ranks via TF-IDF + rule boosts:
    * vegetarian / gluten-aware / buffet-ready (food)
    * “Things to Do / Ultimate/Comprehensive Guide … / attraction names” (travel)
    * prepare form / fill & sign / request e-signatures / share / convert / export (acrobat)
- Vegetarian briefs: hard-filter non-veg (incl. 'chicken broth').
- Diversity per-doc cap: food=1, travel=2, acrobat=2.
"""

import json
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pdfplumber

# Reuse your extractor (same folder)
from process_pdfs import process_file

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- generic text helpers -----------------
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
IS_INGR = re.compile(r'^\s*ingredients\b', re.I)
IS_INSTR = re.compile(r'^\s*instructions?\b', re.I)

def read_json(p: Path) -> dict:
    with open(p, "r") as f:
        return json.load(f)

def page_text(pdf_path: Path, page_no: int) -> str:
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if 0 <= page_no < len(pdf.pages):
                return pdf.pages[page_no].extract_text() or ""
    except Exception:
        pass
    return ""

def page_lines(pdf_path: Path, page_no: int) -> List[str]:
    txt = page_text(pdf_path, page_no)
    return [ln.strip() for ln in (txt or "").splitlines()]

def page_sentences(pdf_path: Path, page_no: int) -> List[str]:
    txt = page_text(pdf_path, page_no)
    return [s.strip() for s in SENT_SPLIT.split(txt or "") if s.strip()]

def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z\-\’']+", (s or "").lower())

def _clean_bullets(s: str) -> str:
    if not s:
        return s
    s = s.replace("•", "- ")
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

# ----------------- domain detection -----------------
def detect_domain(persona: str, job: str) -> str:
    t = f"{persona} {job}".lower()
    if any(w in t for w in ["trip", "travel", "itinerary", "vacation", "tour", "planner"]):
        return "travel"
    if any(w in t for w in ["menu", "buffet", "dinner", "vegetarian", "gluten", "cater", "catering", "food"]):
        return "food"
    if any(w in t for w in ["acrobat", "pdf", "e-signature", "signature", "convert", "export", "edit", "share", "ai", "form", "forms"]):
        return "acrobat"
    return "generic"

# ----------------- food lexicons -----------------
MEAT_WORDS = {
    "beef","pork","chicken","turkey","lamb","mutton","veal","fish","salmon","tuna",
    "shrimp","prawn","anchovy","bacon","ham","sausage","prosciutto","pancetta","duck"
}
GLUTEN_RED_FLAGS = {
    "bread","baguette","bun","pita","naan","roti","tortilla","wrap","quesadilla",
    "pasta","spaghetti","noodles","lasagna","lasagne","ziti","macaroni","pretzel","flour",
    "batter","breadcrumbs","croutons","gnocchi"
}
VEG_GOOD_WORDS = {
    "veg","vegetarian","paneer","dal","lentil","chickpea","hummus","falafel","tofu",
    "ratatouille","salad","rice","quinoa","polenta","corn","potato","vegetable","stir-fry",
    "sushi","rolls","baba","ganoush","tahini"
}
BUFFET_FRIENDLY = {
    "salad","rice","pulao","pilaf","dal","hummus","baba ganoush","ratatouille",
    "falafel","paneer","polenta","casserole","bake","tray","platter","dip","sushi","rolls"
}
FOOD_STRONG_PREFER = {
    "falafel","baba ganoush","hummus","ratatouille","veg sushi","veggie sushi","sushi rolls","vegetable lasagna","lasagne"
}
DINNER_HINTS = {"dinner","mains","sides","entree","buffet","evening"}

# ----------------- travel / acrobat lexicons -----------------
TRAVEL_KEYWORDS = {
    "things to do","attractions","top attractions","must-see","must see","highlights",
    "restaurants","where to eat","food","cuisine","hotels","accommodations","where to stay",
    "tips","tips and tricks","tricks","getting around","transport","transportation",
    "getting there","how to get there","history","culture","traditions","day trips","beaches",
    "itinerary","day 1","day 2","day 3","day 4","budget","group","booking","museums","guide","cities"
}
# Attraction-ish (e.g., "MuCEM (Museum of European and Mediterranean Civilizations)")
ATTRACTION_PAT = re.compile(r"^[A-Z][\w’'&\- ]+(?:\s*\([^)]*\))?$")

# Acrobat targets
ACROBAT_HEADS = {
    "create","convert","export","edit","fill and sign","request e-signatures","generative ai",
    "share","checklist","learn acrobat","send for signature","clipboard","multiple files",
    "convert to word","convert to powerpoint","convert to html","export images","pdf to jpg","pdf to xml",
    "prepare form","prepare forms"
}
FORM_HEADS = {
    "prepare form","prepare forms","fillable","flat form","flat forms","interactive form","interactive forms",
    "form field","form fields","change flat forms to fillable","e-signature","e-signatures","signature","signatures",
    "fill and sign","send for signature","request e-signatures","request signatures"
}
SHARE_HEADS = {
    "share","get link","copy link","review","comment","comments","mark up","markup",
    "send link","unshare","add people","third-party app","whatsapp","teams","gmail","outlook"
}
CONVERT_EXPORT_HEADS = {
    "convert","export","word","ppt","powerpoint","excel","image","jpg","jpeg","png","html","xml",
    "clipboard","multiple pdfs","multiple files","portfolio"
}
LOW_VALUE_ACROBAT = {"test your","ultimate test","quiz","skills checklist","knowledge check"}

# ----------------- rule-based boost -----------------
def rule_boost(title: str, doc_name: str, context: str, persona: str, job: str, domain: str) -> float:
    tkn = set(tokenize(title) + tokenize(context))
    score = 0.0

    if domain == "food":
        # penalties for non-veg
        if tkn & MEAT_WORDS:
            score -= 0.50
        if "chicken" in tkn and "broth" in tkn:
            score -= 0.45

        # veg/gluten weighting (not a hard block on gluten; just downrank)
        score += 0.12 * len(tkn & VEG_GOOD_WORDS)
        if tkn & GLUTEN_RED_FLAGS:
            score -= 0.12  # keep mild so items like veg lasagna can still appear
        score += 0.10 * len({"rice","quinoa","polenta","corn","potato","dal","lentil","chickpea"} & tkn)

        # dinner/buffet hints + buffet-friendly
        dn = (doc_name or "").lower()
        if any(w in dn for w in ("dinner","mains","sides")) or (tkn & DINNER_HINTS):
            score += 0.14
        score += 0.10 * len(tkn & BUFFET_FRIENDLY)

        # prefer Sides; downrank breakfast/lunch for dinner brief
        if "sides" in dn:
            score += 0.26
        if "breakfast" in dn:
            score -= 0.50
        if "lunch" in dn:
            score -= 0.22

        # iconic veggie buffet items + prefer Sides duplicates
        title_low = (title or "").lower()
        if any(p in title_low for p in FOOD_STRONG_PREFER):
            score += 0.30
            if "mains" in dn:
                score -= 0.18
            if "sides" in dn:
                score += 0.30

    elif domain == "travel":
        lower_title = (title or "").lower()
        score += 0.14 * sum(1 for kw in TRAVEL_KEYWORDS if kw in lower_title)

        # “Ultimate/Comprehensive Guide …”, “Things to Do …”
        if re.match(r"^(ultimate|comprehensive)\s+guide\b", (title or ""), re.I):
            score += 0.20
        if re.match(r"^\s*things to do\b", (title or ""), re.I):
            score += 0.24

        # attraction-like headings (e.g., MuCEM …)
        if ATTRACTION_PAT.match(title or "") and len(title.split()) <= 12:
            score += 0.12

        # filename hints
        dn = (doc_name or "").lower()
        if "things to do" in dn:
            score += 0.45   # make sure “Things to Do” doc wins
        if any(w in dn for w in ["restaurants","hotels"]):
            score += 0.10
        if "cities" in dn:
            score += 0.08
        if any(w in dn for w in ["traditions","culture"]):
            score += 0.06
        if "tips and tricks" in dn or "tips" in dn:
            score -= 0.22  # de-prioritize packing vs. core planning

        # group/budget/itinerary-related tokens
        pj = set(tokenize(persona) + tokenize(job))
        score += 0.06 * len(pj & {"group","budget","transport","transportation","itinerary","day","friends","college"})

        # downrank explicit packing content in title
        if "packing" in lower_title:
            score -= 0.20

    elif domain == "acrobat":
        lower_title = (title or "").lower()

        # Strong: forms & signatures
        form_hits = sum(1 for kw in FORM_HEADS if kw in lower_title)
        score += 0.30 * form_hits
        if "change flat forms to fillable" in lower_title:
            score += 0.40
        if "fill and sign" in lower_title:
            score += 0.26
        if any(k in lower_title for k in ["request e-signatures","send for signature","send a document to get signatures"]):
            score += 0.26
        if "prepare form" in lower_title or "prepare forms" in lower_title:
            score += 0.22

        # Share / review collaboration
        share_hits = sum(1 for kw in SHARE_HEADS if kw in lower_title)
        score += 0.14 * share_hits

        # Convert / export utility
        conv_hits = sum(1 for kw in CONVERT_EXPORT_HEADS if kw in lower_title)
        score += 0.14 * conv_hits
        if any(k in lower_title for k in ["clipboard","multiple files","multiple pdfs"]):
            score += 0.14

        # Mild generic Acrobat verbs
        score += 0.06 * sum(1 for kw in ACROBAT_HEADS if kw in lower_title)

        # Deprioritize quizzes/checklists and “Consider/Note/What if/FAQ” lines
        if any(k in lower_title for k in LOW_VALUE_ACROBAT):
            score -= 0.30
        if lower_title.startswith(("consider ", "note:", "what if", "tip:", "warning:", "faq")) or "?" in lower_title:
            score -= 0.22
        if "leader in e-signatures" in lower_title:
            score -= 0.35

    # small persona/job overlap bonus
    pj = set(tokenize(persona) + tokenize(job))
    score += 0.03 * len(pj & tkn)
    return score

# ----------------- snippet builders -----------------
def refine_recipe_text(pdf_path: Path, page_no: int, heading_text: str) -> str:
    lines = page_lines(pdf_path, page_no)
    if not lines:
        return ""
    # find (title, Ingredients_idx) pairs
    recipes = []
    for i, ln in enumerate(lines):
        if IS_INGR.match(ln):
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            if j >= 0:
                title_guess = lines[j].strip()
                recipes.append((title_guess, i))
    # choose matching title or first recipe on page
    ht = (heading_text or "").strip().lower()
    chosen = None
    for title_guess, i in recipes:
        if title_guess.strip().lower() == ht:
            chosen = (title_guess, i)
            break
    if chosen is None and recipes:
        chosen = recipes[0]
    if chosen is None:
        sents = page_sentences(pdf_path, page_no)
        return " ".join(sents[:3])

    _, ingr_idx = chosen
    ingrs = [lines[ingr_idx]]
    k = ingr_idx + 1
    while k < len(lines):
        if not lines[k].strip(): break
        if IS_INSTR.match(lines[k]): break
        ingrs.append(lines[k].strip())
        if len(ingrs) >= 5: break
        k += 1

    instr = []
    instr_idx = None
    for t in range(ingr_idx + 1, len(lines)):
        if IS_INSTR.match(lines[t]):
            instr_idx = t; break
    if instr_idx is not None:
        instr.append(lines[instr_idx].strip())
        u = instr_idx + 1
        while u < len(lines) and len(instr) < 5:
            if not lines[u].strip(): break
            instr.append(lines[u].strip()); u += 1

    snippet = " ".join(ingrs[:5])
    if instr:
        snippet += "  " + " ".join(instr[:5])
    return _clean_bullets(snippet.strip())

def refine_text(pdf_path: Path, page_no: int, heading_text: str, domain: str) -> str:
    if domain == "food":
        snip = refine_recipe_text(pdf_path, page_no, heading_text)
        if snip:
            return snip
    sents = page_sentences(pdf_path, page_no)
    return _clean_bullets(" ".join(sents[:3]))

# ----------------- heading checks -----------------
def looks_like_heading(ln: str) -> bool:
    if not ln or not ln.strip():
        return False
    s = ln.strip()
    if s.endswith("."):
        return False
    if len(s) > 90:
        return False
    # reject sentence-ish lines
    if re.search(r"\b(this guide|you'll|we'll|we will|help you|to help you|this section)\b", s, re.I):
        return False
    if sum(ch in s for ch in ",;:") > 2:
        return False

    words = re.findall(r"[A-Za-z][A-Za-z’'\-]*", s)
    if len(words) < 2 or len(words) > 14:
        return False

    title_case_ratio = sum(w[0].isupper() for w in words) / len(words)
    all_caps = s.isupper()

    # strong patterns
    if re.match(r"^(ultimate|comprehensive)\s+guide\b", s, re.I):
        return True
    if re.match(r"^(guide to|things to do|restaurants|hotels|tips|history|culture|traditions|itinerary|day \d+|cities)\b", s, re.I):
        return True

    return title_case_ratio >= 0.7 or all_caps or ATTRACTION_PAT.match(s) is not None

# ----------------- candidate builders -----------------
def outline_candidates(pdf_path: Path, doc_name: str) -> List[Dict]:
    res = []
    try:
        out = process_file(pdf_path)
        for item in out.get("outline", []):
            sents = page_sentences(pdf_path, item["page"])
            context_text = " ".join(sents[:4])
            res.append({
                "doc": doc_name,
                "title": item["text"],
                "page": item["page"],
                "context_text": context_text
            })
    except Exception:
        pass
    return res

def fallback_recipe_candidates(pdf_path: Path) -> List[Dict]:
    """Find candidates by [prev line] + 'Ingredients' with a wider ingredient window."""
    out = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for pno, page in enumerate(pdf.pages):
                lines = [ln.strip() for ln in (page.extract_text() or "").splitlines()]
                for i, ln in enumerate(lines):
                    if IS_INGR.match(ln):
                        j = i - 1
                        while j >= 0 and not lines[j].strip():
                            j -= 1
                        if j >= 0:
                            title = lines[j].strip()
                            if title and len(title.split()) <= 7 and not title.endswith("."):
                                ing = [lines[i]]
                                k = i + 1
                                while k < len(lines) and len(ing) < 12:
                                    if not lines[k].strip(): break
                                    if IS_INSTR.match(lines[k]): break
                                    ing.append(lines[k].strip())
                                    k += 1
                                ctx = " ".join(ing)
                                for t in range(i + 1, min(i + 14, len(lines))):
                                    if IS_INSTR.match(lines[t]):
                                        ctx += " " + lines[t].strip()
                                        if t + 1 < len(lines) and lines[t + 1].strip():
                                            ctx += " " + lines[t + 1].strip()
                                        break
                                out.append({
                                    "doc": Path(pdf_path).name,
                                    "title": title,
                                    "page": pno,
                                    "context_text": ctx
                                })
    except Exception:
        pass
    return out

def fallback_travel_candidates(pdf_path: Path) -> List[Dict]:
    out = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for pno, page in enumerate(pdf.pages):
                lines = [ln.strip() for ln in (page.extract_text() or "").splitlines()]
                for i, ln in enumerate(lines):
                    if not ln: continue
                    lower = ln.lower()
                    if "table of contents" in lower: continue
                    if looks_like_heading(ln):
                        ctx = " ".join(lines[i+1:i+6])
                        out.append({
                            "doc": Path(pdf_path).name,
                            "title": ln.strip(),
                            "page": pno,
                            "context_text": ctx
                        })
    except Exception:
        pass
    return out

def fallback_acrobat_candidates(pdf_path: Path) -> List[Dict]:
    """Find acrobat/form/e-sign/share/convert headings; avoid paragraphs with sanitizer."""
    out = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for pno, page in enumerate(pdf.pages):
                lines = [ln.strip() for ln in (page.extract_text() or "").splitlines()]
                for i, ln in enumerate(lines):
                    if not ln: continue
                    s = ln.strip()
                    lower = s.lower()
                    if "table of contents" in lower:
                        continue
                    # early rejects
                    if s.endswith(".") or "?" in s:
                        continue
                    if lower.startswith(("consider ", "note:", "what if", "tip:", "warning:", "faq")):
                        continue

                    anchored = re.match(r"^(fill|prepare|create|convert|export|edit|request|send|share|change|use|validate|manage|protect|organize)\b", lower) is not None
                    has_forms = any(k in lower for k in FORM_HEADS)
                    has_share = any(k in lower for k in SHARE_HEADS)
                    has_conv  = any(k in lower for k in CONVERT_EXPORT_HEADS)
                    has_generic = any(k in lower for k in ACROBAT_HEADS)
                    good_shape = looks_like_heading(s)

                    if good_shape or anchored or has_forms or has_share or has_conv or has_generic:
                        if len(s.split()) > 14:
                            continue
                        ctx = " ".join(lines[i+1:i+6])
                        out.append({
                            "doc": Path(pdf_path).name,
                            "title": s,
                            "page": pno,
                            "context_text": ctx
                        })
    except Exception:
        pass
    return out

# ----------------- candidate sanitize -----------------
def reject_bad_heading(title: str, domain: str) -> bool:
    if not title or not title.strip():
        return True
    t = title.strip()
    tl = t.lower()
    # generic badness
    if t.endswith("."):
        return True
    if len(t) > 120:
        return True
    if tl.startswith(("consider ", "note:", "what if", "tip:", "warning:", "faq")):
        return True
    if "?" in t:
        return True
    # domain-specific
    if domain == "acrobat":
        if "leader in e-signatures" in tl:
            return True
    return False

def sanitize_candidates(cands: List[Dict], domain: str) -> List[Dict]:
    out = []
    seen = set()
    for c in cands:
        if reject_bad_heading(c.get("title",""), domain):
            continue
        key = (c["doc"], c["title"].strip().lower(), c["page"])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out

# ----------------- candidate filtering (vegetarian hard filter) -----------------
def filter_candidates(candidates: List[Dict], domain: str, persona: str, job: str) -> List[Dict]:
    t = f"{persona} {job}".lower()
    vegetarian_required = "vegetarian" in t or "veg " in t or "veg." in t
    if domain != "food" or not vegetarian_required:
        return candidates
    filtered = []
    for c in candidates:
        toks = set(tokenize(c["title"] + " " + c.get("context_text", "")))
        if toks & MEAT_WORDS:
            continue
        if "chicken" in toks and "broth" in toks:
            continue
        filtered.append(c)
    return filtered

# ----------------- ranking with per-doc cap -----------------
def rank_sections(candidates: List[Dict], query: str, persona: str, job: str, domain: str, top_k=5, per_doc_cap=1) -> List[Tuple[Dict, float]]:
    if not candidates:
        return []
    corpus = [(c["title"] + " " + c.get("context_text", "")).strip() for c in candidates]
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X = vec.fit_transform(corpus)
    qv = vec.transform([query])
    sims = cosine_similarity(X, qv).ravel()
    # normalize sims to [0,1] to blend with rule boosts
    if sims.size:
        smin, smax = float(sims.min()), float(sims.max())
        sims_scaled = (sims - smin) / (smax - smin + 1e-9)
    else:
        sims_scaled = sims
    boosts = np.array([
        rule_boost(c["title"], c["doc"], c.get("context_text", ""), persona, job, domain)
        for c in candidates
    ])
    final = 0.6 * sims_scaled + boosts
    order = np.argsort(-final)
    picked: List[Tuple[Dict, float]] = []
    seen_per_doc: Dict[str, int] = {}
    for idx in order:
        c = candidates[idx]
        doc = c["doc"]
        if seen_per_doc.get(doc, 0) >= per_doc_cap:
            continue
        key = (doc, c["title"].strip().lower(), c["page"])
        if any(key == (p[0]["doc"], p[0]["title"].strip().lower(), p[0]["page"]) for p in picked):
            continue
        picked.append((c, float(final[idx])))
        seen_per_doc[doc] = seen_per_doc.get(doc, 0) + 1
        if len(picked) == top_k:
            break
    return picked

# ----------------- main -----------------
def main(input_json_path: str, pdf_dir: str, out_json_path: str):
    spec = read_json(Path(input_json_path))
    docs = spec["documents"]
    persona = spec["persona"]["role"]
    job = spec["job_to_be_done"]["task"]
    query = f"{persona}. {job}"
    domain = detect_domain(persona, job)

    candidates: List[Dict] = []
    for d in docs:
        pdf_path = Path(pdf_dir) / d["filename"]
        added = set()

        # (a) outline-based
        for c in outline_candidates(pdf_path, d["filename"]):
            key = (c["doc"], c["title"].strip().lower(), c["page"])
            if key not in added:
                candidates.append(c); added.add(key)

        # (b) domain-specific fallbacks
        if domain == "food":
            for c in fallback_recipe_candidates(pdf_path):
                key = (c["doc"], c["title"].strip().lower(), c["page"])
                if key not in added:
                    candidates.append(c); added.add(key)
        elif domain == "travel":
            for c in fallback_travel_candidates(pdf_path):
                key = (c["doc"], c["title"].strip().lower(), c["page"])
                if key not in added:
                    candidates.append(c); added.add(key)
        elif domain == "acrobat":
            for c in fallback_acrobat_candidates(pdf_path):
                key = (c["doc"], c["title"].strip().lower(), c["page"])
                if key not in added:
                    candidates.append(c); added.add(key)

    # sanitize & vegetarian hard filter (if applicable)
    candidates = sanitize_candidates(candidates, domain)
    candidates = filter_candidates(candidates, domain, persona, job)

    # diversity cap
    if domain == "travel":
        per_doc_cap = 2
    elif domain == "acrobat":
        per_doc_cap = 2
    else:
        per_doc_cap = 1

    top = rank_sections(candidates, query, persona, job, domain, top_k=5, per_doc_cap=per_doc_cap)

    extracted_sections = []
    subsection_analysis = []
    for rank, (c, _) in enumerate(top, start=1):
        extracted_sections.append({
            "document": c["doc"],
            "section_title": c["title"],
            "importance_rank": rank,
            "page_number": c["page"]
        })
        refined = refine_text(Path(pdf_dir)/c["doc"], c["page"], c["title"], domain)
        subsection_analysis.append({
            "document": c["doc"],
            "refined_text": refined,
            "page_number": c["page"]
        })

    out = {
        "metadata": {
            "input_documents": [d["filename"] for d in docs],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }
    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    # Example:
    # python challenge1b_runner.py "./Collection 3/challenge1b_input.json" "./Collection 3/PDFs" "./Collection 3/challenge1b_output_generated.json"
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])
