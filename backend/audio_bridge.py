# audio_bridge.py
import os, json, uuid, re
from pathlib import Path

import requests  # make sure requests is in your image

APP_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_ROOT", APP_DIR / "data")).resolve()
AUDIO_DIR = DATA_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

def _pick_ext(fmt: str) -> str:
    f = (fmt or "").lower()
    if "wav" in f: return ".wav"
    if "pcm" in f: return ".pcm"
    if "ogg" in f or "opus" in f: return ".ogg"
    if "aac" in f: return ".aac"
    if "flac" in f: return ".flac"
    return ".mp3"

def _openai_tts_format(fmt: str) -> str:
    """Map arbitrary format strings to OpenAI audio.speech 'format' values."""
    f = (fmt or "").lower()
    if "wav" in f: return "wav"
    if "ogg" in f or "opus" in f: return "ogg"
    if "aac" in f: return "aac"
    if "flac" in f: return "flac"
    if "pcm" in f: return "pcm"
    return "mp3"  # default

def synthesize(text: str, *, voice: str = None, fmt: str = "audio-48khz-192kbitrate-mono-mp3") -> Path:
    provider = (os.getenv("TTS_PROVIDER", "azure") or "").lower()
    if provider == "azure":
        return _synthesize_azure_openai(text, voice=voice, fmt=fmt)
    raise RuntimeError(f"Unsupported TTS_PROVIDER={provider}. Set TTS_PROVIDER=azure.")

def _synthesize_azure_openai(text: str, *, voice: str = None, fmt: str = "audio-48khz-192kbitrate-mono-mp3") -> Path:
    endpoint = (os.getenv("AZURE_TTS_ENDPOINT") or "").strip()
    key = (os.getenv("AZURE_TTS_KEY") or "").strip()
    if not endpoint or not key:
        raise RuntimeError("Azure OpenAI TTS misconfigured: missing AZURE_TTS_ENDPOINT or AZURE_TTS_KEY")

    # Use OpenAI TTS voice names (e.g., 'alloy', 'verse'), not Azure Speech names.
    voice = (voice or os.getenv("AZURE_TTS_VOICE") or "alloy").strip()

    # Deduce model from the endpoint (deployments/{name}/audio/speech) if possible
    m = re.search(r"/deployments/([^/]+)/audio/speech", endpoint)
    model = m.group(1) if m else (os.getenv("AZURE_TTS_MODEL") or "gpt-4o-mini-tts-2")

    outfmt = _openai_tts_format(fmt)
    payload = {"model": model, "input": text, "voice": voice, "format": outfmt}
    headers = {"api-key": key, "content-type": "application/json"}

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        if resp.status_code >= 400:
            detail = resp.text
            raise RuntimeError(f"Azure OpenAI TTS failed: {resp.status_code} {resp.reason}; body={detail[:300]}")
        audio_bytes = resp.content
        if not audio_bytes:
            raise RuntimeError("Empty audio from Azure OpenAI TTS")

        ext = _pick_ext(outfmt)
        out = AUDIO_DIR / f"tts_{uuid.uuid4().hex}{ext}"
        out.write_bytes(audio_bytes)
        return out
    except Exception as e:
        print("[TTS ERROR]", repr(e))
        raise RuntimeError(f"Azure OpenAI TTS failed: {e}")