# -*- coding: utf-8 -*-
# app/rag/retriever.py

from __future__ import annotations

from pathlib import Path
import json
import re
import yaml

import faiss
from sentence_transformers import SentenceTransformer

from app.rag.utils_text import safe_truncate

# اقرأ config.yaml مباشرة (تجنب مشكلة app.config)
with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f) or {}

EMBED_MODEL = (CFG.get("models") or {}).get(
    "embeddings",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
EMB = SentenceTransformer(EMBED_MODEL)

IDX_PATH = Path("app/index/faiss_hnsw.index")
METAS_PATH = Path("app/index/metas.json")

if not IDX_PATH.exists():
    raise FileNotFoundError(f"FAISS index not found: {IDX_PATH}")
if not METAS_PATH.exists():
    raise FileNotFoundError(f"metas.json not found: {METAS_PATH}")

_index = faiss.read_index(str(IDX_PATH))
_metas = json.loads(METAS_PATH.read_text(encoding="utf-8"))

_PMID_RE = re.compile(r"\bPMID:\s*(\d{6,9})\b")
_DOI_RE  = re.compile(r"\bdoi:\s*([0-9.]+/[^\s]+)", re.IGNORECASE)

def _extract_ids(text: str):
    if not text:
        return None, None
    pm = _PMID_RE.search(text)
    doi = _DOI_RE.search(text)
    return (pm.group(1) if pm else None, doi.group(1) if doi else None)

def _to_similarity(dist_or_score: float) -> float:
    """
    خلي score الأعلى أفضل:
    - METRIC_L2 مع embeddings normalized: cos = 1 - D/2
    - METRIC_INNER_PRODUCT: كما هو
    """
    metric = getattr(_index, "metric_type", None)
    if metric == faiss.METRIC_L2:
        return float(1.0 - (dist_or_score / 2.0))
    return float(dist_or_score)

def hybrid_search(query: str, top_k: int | None = None):
    q = (query or "").strip()
    if not q:
        return []

    k = top_k or int((CFG.get("retrieval") or {}).get("top_k_merged", 5))

    q_vec = EMB.encode([q], normalize_embeddings=True).astype("float32")
    D, I = _index.search(q_vec, k)

    results = []
    for idx, dist_or_score in zip(I[0], D[0]):
        if int(idx) < 0 or int(idx) >= len(_metas):
            continue

        meta = _metas[int(idx)]
        sim = _to_similarity(float(dist_or_score))

        pmid, doi = _extract_ids(meta.get("text", "") or "")

        results.append({
            "title": meta.get("title", ""),
            "source": meta.get("source", ""),
            "pmid": pmid,
            "doi": doi,
            "snippet": safe_truncate(meta.get("text", ""), 350),
            "score": sim,
            "chunk_id": meta.get("chunk_id", meta.get("id", int(idx))),
        })

    results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return results

# alias
search = hybrid_search
