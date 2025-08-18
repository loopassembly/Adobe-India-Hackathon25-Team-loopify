# scoring.py
import math
from typing import Optional

def coerce_score_percent(raw: Optional[float]) -> float:
    """
    Normalize a raw similarity score to [0,100] for UI.
    - If raw is in [-1,1] (cosine/IP), map to [0,100] via ((s+1)/2)*100.
    - If raw seems already in [0,1], scale by 100.
    - Non-finite/None -> 0.
    """
    if raw is None:
        return 0.0
    try:
        s = float(raw)
    except Exception:
        return 0.0
    if not math.isfinite(s):
        return 0.0

    # Clamp for stability
    if -1.0 <= s <= 1.0:
        pct = (s + 1.0) * 50.0
    elif 0.0 <= s <= 1.0:
        pct = s * 100.0
    else:
        # Fallback: squash via tanh then map
        s = max(-5.0, min(5.0, s))
        pct = (math.tanh(s) + 1.0) * 50.0

    return round(max(0.0, min(100.0, pct)), 1)
