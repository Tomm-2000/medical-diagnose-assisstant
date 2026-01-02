# -*- coding: utf-8 -*-
# app/rag/build_index.py

import os
import json

import faiss
import numpy as np  # مو ضروري حالياً بس منخليه
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv
import torch
import yaml

# نقرأ config.yaml من جذر المشروع
with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

load_dotenv(find_dotenv(usecwd=True), encoding="utf-8-sig", override=True)

# اسم نموذج الـ embeddings من الـ config
EMBED = CFG.get("models", {}).get(
    "embeddings",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)


def load_chunks(d="app/kb/chunks"):
    metas = []
    texts = []
    for fname in os.listdir(d):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(d, fname)
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        metas.append(rec)
        texts.append(rec["text"])
    return metas, texts


if __name__ == "__main__":
    os.makedirs("app/index", exist_ok=True)
    metas, texts = load_chunks()
    if not texts:
        raise SystemExit("لا توجد مقاطع. شغّل prepare_kb أولاً.")

    # اختيار الجهاز: GPU إذا متوفر، غير هيك CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device for embeddings:", device)

    # تحميل نموذج الـ embeddings على الـ GPU أو CPU حسب المتوفر
    model = SentenceTransformer(EMBED, device=device)

    # الترميز
    X = model.encode(texts, normalize_embeddings=True).astype("float32")

    # بناء الـ index
    idx = faiss.index_factory(X.shape[1], "HNSW32,Flat")
    idx.add(X)

    # حفظ الإندكس والميتا
    faiss.write_index(idx, "app/index/faiss_hnsw.index")
    with open("app/index/metas.json", "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False)

    print(f"index size = {len(metas)}")
