# -*- coding: utf-8 -*-
# app/rag/add_pubmed_stroke_chunks.py

import json
import re
import yaml
from pathlib import Path
from tqdm import tqdm

# Load config to get chunk size/overlap
with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f) or {}

KB_CFG = CFG.get("kb", {})
CHUNK_SIZE = KB_CFG.get("chunk_size", 1000)
CHUNK_OVERLAP = KB_CFG.get("chunk_overlap", 200)

INPUT_FILE = Path("data/pubmed_subset/pubmed_subset.jsonl")
CHUNKS_DIR = Path("app/kb/chunks")
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Stroke detection ----------
STROKE_PATTERNS = [
    r"\bstroke\b",
    r"\bcva\b",
    r"\bbrain attack\b",
    r"\bcerebrovascular accident\b",
    r"\bischemic stroke\b",
    r"\bhemorrhagic stroke\b",
    r"\bintracerebral hemorrhage\b",
    r"\bsubarachnoid hemorrhage\b",
    r"\bTIA\b",
    r"\btransient ischemic attack\b",
    r"\bcerebral infarction\b",
    r"\bbrain infarction\b",
    r"\bthrombectomy\b",
    r"\bthrombolysis\b",
    r"\bpost-stroke\b",
]
STROKE_RE = re.compile("|".join(STROKE_PATTERNS), re.IGNORECASE)

# ---------- Posterior circulation keywords ----------
POSTERIOR_KEYWORDS = [
    r"\bvertebrobasilar\b",
    r"\bposterior circulation\b",
    r"\bbrainstem\b",
    r"\bcerebellar\b",
    r"\bvertigo\b",
    r"\bdiplopia\b",
    r"\bataxia\b",
    r"\bdizziness\b",
    r"\bcross sensory\b",
    r"\balternating\b",
    r"\bhoarseness\b",
    r"\bdysphagia\b",
]
POSTERIOR_RE = re.compile("|".join(POSTERIOR_KEYWORDS), re.IGNORECASE)

# ---------- Chunking ----------
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)
        chunks.append(text[i:j])
        if j >= n:
            break
        i = j - overlap
    return chunks

def main():
    if not INPUT_FILE.exists():
        raise SystemExit(f"❌ PubMed subset file not found: {INPUT_FILE}")

    # Remove old chunks
    for f in CHUNKS_DIR.glob("*.json"):
        f.unlink()

    total = 0
    kept = 0
    chunk_count = 0

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scanning PubMed subset"):
            total += 1
            try:
                rec = json.loads(line)
            except:
                continue

            title = rec.get("title", "")
            text = rec.get("text", "")
            pmid = rec.get("pmid")

            full_text = f"{title}\n{text}".strip()

            # نحتفظ بالمقالة إذا احتوت على مصطلحات السكتة أو الدورة الخلفية
            if not (STROKE_RE.search(full_text) or POSTERIOR_RE.search(full_text)):
                continue
            if len(text.strip()) < 80:
                continue

            kept += 1
            chunks = chunk_text(full_text)

            for i, ch in enumerate(chunks, start=1):
                chunk_id = f"pmid_{pmid}_chunk_{i}"
                out = {
                    "id": chunk_id,
                    "chunk_id": chunk_id,
                    "pmid": pmid,
                    "title": title,
                    "text": ch,
                    "source": "PubMedSubset"
                }
                (CHUNKS_DIR / f"{chunk_id}.json").write_text(
                    json.dumps(out, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                chunk_count += 1

    print("\n==============================")
    print(f"Total scanned: {total:,}")
    print(f"Stroke records kept: {kept:,}")
    print(f"Total chunks written: {chunk_count:,}")
    print("Chunks directory:", CHUNKS_DIR)
    print("==============================")

if __name__ == "__main__":
    main()
