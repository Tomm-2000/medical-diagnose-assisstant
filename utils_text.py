import re

def squeeze_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def safe_truncate(s: str, n: int = 800) -> str:
    s = s.strip()
    if len(s) <= n:
        return s
    cut = s[:n]
    # try to cut at sentence boundary
    m = re.search(r"[.!?]\s+\S*$", cut)
    if m:
        cut = cut[:m.start()+1]
    return cut + " …"
