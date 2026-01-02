# -*- coding: utf-8 -*-
# app/rag/add_stroke_chunks.py

import os
import json
from pathlib import Path

# مسار ملف الـ abstracts المنظّف
RAW_FILE = Path("data/raw/pubmed_stroke_abstracts_clean.txt")

# مجلد الـ KB اللي بيقرأ منو build_index.py
KB_DIR = Path("app/kb/chunks")
KB_DIR.mkdir(parents=True, exist_ok=True)


def simple_chunk(text: str, chunk_size: int = 700, overlap: int = 150):
    """تقطيع النص لقطع متداخلة شوي (للاسترجاع)."""
    if chunk_size <= 0:
        chunk_size = 700
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size - 1

    chunks = []
    n = len(text)
    i = 0
    while i < n:
        end = min(i + chunk_size, n)
        chunks.append(text[i:end])
        if end >= n:
            break
        i = end - overlap
    return chunks


def main():
    if not RAW_FILE.exists():
        raise SystemExit(f"❌ الملف غير موجود: {RAW_FILE}")

    text = RAW_FILE.read_text(encoding="utf-8")
    text = text.strip()
    if not text:
        raise SystemExit("❌ ملف الـ abstracts فارغ بعد التنظيف.")

    chunks = simple_chunk(text, chunk_size=700, overlap=150)
    print(f"عدد الـ chunks الناتجة من النص: {len(chunks)}")

    # نكتب كل chunk كـ ملف JSON منفصل في app/kb/chunks
    for i, ch in enumerate(chunks, start=1):
        rec = {
            "id": f"stroke_local_{i}",
            "title": "pubmed_stroke_abstracts",
            "text": ch,
            "source": str(RAW_FILE),
        }
        out_path = KB_DIR / f"stroke_local_{i}.json"
        out_path.write_text(
            json.dumps(rec, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"✅ تم حفظ {len(chunks)} ملف JSON في: {KB_DIR}")


if __name__ == "__main__":
    main()
