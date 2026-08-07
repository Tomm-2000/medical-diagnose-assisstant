# app/models/medcpt_reranker.py
from __future__ import annotations

from typing import List, Dict, Optional
import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


_DEFAULT_MODEL = os.getenv("MEDRAG_MEDCPT_RERANKER", "ncbi/MedCPT-Cross-Encoder")


class MedCPTReranker:
    """
    Cross-encoder reranker:
    يعطي score لكل (query, doc_text) وبيرتّب النتائج.
    """
    def __init__(self, model_name: str = _DEFAULT_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name,use_safetensors=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, query: str, docs: List[str], batch_size: int = 8, max_length: int = 512) -> List[float]:
        scores: List[float] = []
        q = (query or "").strip()

        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            pairs = [(q, (d or "")) for d in batch_docs]

            enc = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            out = self.model(**enc)
            logits = out.logits

            # غالباً shape [B,1] أو [B,2] حسب head
            if logits.dim() == 2 and logits.size(-1) == 1:
                batch_scores = logits.squeeze(-1)
            else:
                # خذ logit class-1 كـ relevance
                batch_scores = logits[:, -1]

            scores.extend(batch_scores.detach().float().cpu().tolist())

        return scores


_RERANKER_SINGLETON: Optional[MedCPTReranker] = None


def get_reranker() -> MedCPTReranker:
    global _RERANKER_SINGLETON
    if _RERANKER_SINGLETON is None:
        _RERANKER_SINGLETON = MedCPTReranker()
    return _RERANKER_SINGLETON


def rerank_docs(query: str, docs: List[Dict], text_key: str = "text") -> List[Dict]:
    """
    docs: list of dicts (مثل merged_docs عندك).
    بيرجع نسخة مرتبة حسب medcpt_score تنازلياً.
    """
    if not docs:
        return docs

    rr = get_reranker()
    texts = [(d.get(text_key) or d.get("snippet") or "") for d in docs]
    scores = rr.score(query, texts)

    out = []
    for d, s in zip(docs, scores):
        d2 = dict(d)
        d2["medcpt_score"] = float(s)
        out.append(d2)

    out.sort(key=lambda x: float(x.get("medcpt_score", -1e9)), reverse=True)
    return out