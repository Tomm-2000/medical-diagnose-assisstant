# app/rag/build_bm25.py
from pathlib import Path
import json
import re
import joblib
from rank_bm25 import BM25Okapi

METAS_PATH = Path("app/index/metas.json")
OUT_PATH = Path("app/index/bm25.joblib")

_WORD_RE = re.compile(r"[A-Za-z0-9]+|[\u0600-\u06FF]+", re.UNICODE)

def tokenize(t: str):
    return _WORD_RE.findall((t or "").lower())

def main():
    if not METAS_PATH.exists():
        raise SystemExit("❌ metas.json غير موجود — شغّل build_index.py أولاً")

    metas = json.loads(METAS_PATH.read_text(encoding="utf-8"))
    corpus = [tokenize(m.get("text") or "") for m in metas]

    bm25 = BM25Okapi(corpus)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"bm25": bm25, "corpus_len": len(metas)}, OUT_PATH)

    print("✅ Saved:", OUT_PATH)
    print("✅ docs:", len(metas))

if __name__ == "__main__":
    main()
