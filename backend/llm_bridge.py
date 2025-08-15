# llm_bridge.py
"""
Tiny wrapper that calls the Adobe-provided chat_with_llm.py in-process if possible,
else via subprocess. The script reads provider/model from ENV, as per finale doc.
"""

import os, subprocess, json, importlib

def generate_insight(prompt: str) -> str:
    # Try in-process import
    try:
        mod = importlib.import_module("chat_with_llm")
        if hasattr(mod, "chat_with_llm"):
            # The sample script’s function typically returns a string
            return mod.chat_with_llm(prompt)  # relies on env: LLM_PROVIDER, GEMINI_MODEL, OPENAI_*
    except Exception:
        pass

    # Fallback: subprocess
    # Many teams implement `python chat_with_llm.py --prompt "..."` that prints the text
    try:
        pr = subprocess.run(
            ["python", "chat_with_llm.py", "--prompt", prompt],
            check=True, capture_output=True, text=True
        )
        out = pr.stdout.strip()
        return out or "No insight generated."
    except Exception as e:
        return f"LLM error: {e}"