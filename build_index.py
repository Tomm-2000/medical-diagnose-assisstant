# -*- coding: utf-8 -*-
# app/rag/build_index.py
# Build FAISS + optional BM25 from chunk JSON files

import os
import json
import re
from pathlib import Path

import faiss
import yaml
import torch
import joblib
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------- Paths ----------
CHUNKS_DIR = Path("app/kb/chunks")

INDEX_DIR = Path("app/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

FAISS_PATH = INDEX_DIR / "faiss_hnsw.index"
METAS_PATH = INDEX_DIR / "metas.json"
BM25_PATH  = INDEX_DIR / "bm25.joblib"

# ---------- Tokenizer ----------
_WORD_RE = re.compile(r"[A-Za-z0-9]+|[\u0600-\u06FF]+", re.UNICODE)

def tokenize(text: str):
    return _WORD_RE.findall((text or "").lower())

# ---------- Load config ----------
with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f) or {}

EMBED_MODEL = (CFG.get("models") or {}).get(
    "embeddings",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

BATCH_SIZE = int(os.getenv("MEDRAG_EMB_BATCH", "256"))
BUILD_BM25 = os.getenv("MEDRAG_BUILD_BM25", "0").strip() == "1"

def load_chunk_files():
    if not CHUNKS_DIR.exists():
        raise SystemExit("❌ chunks directory not found")

    files = list(CHUNKS_DIR.glob("*.json"))
    if not files:
        raise SystemExit("❌ no chunk files found")

    for f in files:
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
            yield rec
        except:
            continue

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    st_model = SentenceTransformer(EMBED_MODEL, device=device)

    metas = []
    bm25_corpus = []
    faiss_index = None

    batch_texts = []
    batch_metas = []

    total = 0

    for rec in tqdm(load_chunk_files(), desc="Reading chunks", unit="chunk"):

        text = rec.get("text", "").strip()
        if not text:
            continue

        batch_texts.append(text)
        batch_metas.append(rec)

        if BUILD_BM25:
            bm25_corpus.append(tokenize(text))

        total += 1

        if len(batch_texts) >= BATCH_SIZE:
            X = st_model.encode(batch_texts, normalize_embeddings=True).astype("float32")

            if faiss_index is None:
                dim = X.shape[1]
                faiss_index = faiss.IndexHNSWFlat(dim, 32)
                faiss_index.hnsw.efConstruction = 200

            faiss_index.add(X)
            metas.extend(batch_metas)

            batch_texts = []
            batch_metas = []

    # flush last batch
    if batch_texts:
        X = st_model.encode(batch_texts, normalize_embeddings=True).astype("float32")
        if faiss_index is None:
            dim = X.shape[1]
            faiss_index = faiss.IndexHNSWFlat(dim, 32)
            faiss_index.hnsw.efConstruction = 200

        faiss_index.add(X)
        metas.extend(batch_metas)

    if faiss_index is None:
        raise SystemExit("❌ nothing indexed")

    faiss.write_index(faiss_index, str(FAISS_PATH))
    METAS_PATH.write_text(json.dumps(metas, ensure_ascii=False), encoding="utf-8")

    print("✅ FAISS saved:", FAISS_PATH)
    print("✅ METAS saved:", METAS_PATH)
    print("✅ Total indexed:", len(metas))

    if BUILD_BM25:
        print("Building BM25...")
        bm25 = BM25Okapi(bm25_corpus)
        joblib.dump({"bm25": bm25, "corpus_len": len(metas)}, BM25_PATH)
        print("✅ BM25 saved:", BM25_PATH)
    else:
        print("ℹ️ BM25 skipped")

if __name__ == "__main__":
    main()
