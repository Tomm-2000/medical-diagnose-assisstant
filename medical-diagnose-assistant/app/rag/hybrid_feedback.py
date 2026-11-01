# -*- coding: utf-8 -*-
"""
app/rag/hybrid_feedback.py

Hybrid pseudo-relevance feedback for MedRAG.

This module is intentionally NOT FAISS-only.
It builds a second-stage retrieval signal from the first-stage evidence coming
from all available retrieval channels:

    1) FAISS dense results
    2) BM25 lexical results
    3) Knowledge Graph article results

The output is:
    - a refined query vector for second-stage FAISS search,
    - an evidence-driven textual feedback query for second-stage BM25/KG search,
    - a diagnostic dictionary for debugging/explainability.

No disease-specific expert expansion templates are used here. The feedback terms
come from the retrieved evidence itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import re

import numpy as np
from app.rag.common_utils import _as_float32_2d


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}|[\u0600-\u06FF]{2,}", re.UNICODE)

# Generic words that should not become expansion terms. Keep this intentionally
# broad and non-disease-specific; this is not an expert template list.
_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "were", "was", "are",
    "has", "had", "have", "not", "but", "all", "can", "may", "into", "our",
    "their", "they", "his", "her", "its", "who", "what", "when", "where", "which",
    "case", "report", "study", "review", "patients", "patient", "clinical", "using",
    "use", "used", "associated", "analysis", "results", "conclusion", "background",
    "objective", "methods", "method", "there", "after", "before", "within", "during",
    "present", "presents", "presented", "presenting", "presentation", "manifest", "manifestation",
    "between", "among", "acute", "chronic", "new", "old", "year", "years", "man", "woman",
    "male", "female", "يلي", "على", "من", "في", "إلى", "عن", "مع", "هذا", "هذه",
}


@dataclass
class HybridFeedbackInfo:
    enabled: bool
    used: bool
    reason: str
    dense_seed_docs: int = 0
    bm25_seed_docs: int = 0
    kg_seed_docs: int = 0
    feedback_docs_used: int = 0
    feedback_terms_used: int = 0
    alpha: float = 0.75
    beta: float = 0.25

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "used": self.used,
            "reason": self.reason,
            "dense_seed_docs": self.dense_seed_docs,
            "bm25_seed_docs": self.bm25_seed_docs,
            "kg_seed_docs": self.kg_seed_docs,
            "feedback_docs_used": self.feedback_docs_used,
            "feedback_terms_used": self.feedback_terms_used,
            "alpha": self.alpha,
            "beta": self.beta,
        }


def l2_normalize(vec: Any, eps: float = 1e-12) -> np.ndarray:
    arr = _as_float32_2d(vec)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (arr / norms).astype("float32")


def search_index_with_vector(index: Any, vector: Any, k: int) -> List[Tuple[int, float]]:
    q_vec = l2_normalize(vector)
    D, I = index.search(q_vec.astype("float32"), int(k))
    return [(int(idx), float(val)) for idx, val in zip(I[0], D[0])]


def _meta_text(meta: Dict[str, Any], max_chars: int) -> str:
    title = str(meta.get("title", "") or "").strip()
    text = str(meta.get("text", "") or meta.get("snippet", "") or "").strip()
    combined = (title + "\n" + text).strip()
    return combined[:max_chars]


def _valid_meta_idx(idx: Any, metas: Sequence[Dict[str, Any]]) -> Optional[int]:
    try:
        i = int(idx)
    except Exception:
        return None
    if 0 <= i < len(metas):
        return i
    return None


def _get_meta_for_pmid(pmid: Any, pmid_to_meta_indices: Dict[str, List[int]], metas: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if pmid is None:
        return None
    indices = pmid_to_meta_indices.get(str(pmid), [])
    for idx in indices:
        i = _valid_meta_idx(idx, metas)
        if i is not None:
            return metas[i]
    return None


def _add_seed_doc(
    docs: List[Dict[str, Any]],
    *,
    source: str,
    text: str,
    weight: float,
    key: str,
) -> None:
    text = (text or "").strip()
    if len(text) < 20:
        return
    weight = float(weight)
    if not math.isfinite(weight) or weight <= 0:
        weight = 0.1
    docs.append({"source": source, "text": text, "weight": weight, "key": key})


def _normalize_scores(pairs: Sequence[Tuple[int, float]], higher_is_better: bool = True) -> Dict[int, float]:
    valid = [(int(i), float(s)) for i, s in pairs if int(i) >= 0]
    if not valid:
        return {}
    vals = [s for _, s in valid]
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-9:
        return {i: 1.0 for i, _ in valid}
    if higher_is_better:
        return {i: (s - mn) / (mx - mn) for i, s in valid}
    return {i: 1.0 - ((s - mn) / (mx - mn)) for i, s in valid}


def _tokenize_for_terms(text: str) -> List[str]:
    toks = [t.lower() for t in _WORD_RE.findall(text or "")]
    return [t for t in toks if t not in _STOPWORDS and len(t) >= 3]


def _extract_phrases(text: str, max_ngram: int = 3) -> Iterable[str]:
    toks = _tokenize_for_terms(text)
    # unigrams + compact bigrams/trigrams. This is evidence-derived, not an
    # expert template; it lets BM25/KG see terms frequent in top evidence.
    for n in range(1, max_ngram + 1):
        if len(toks) < n:
            continue
        for i in range(0, len(toks) - n + 1):
            phrase = " ".join(toks[i:i+n])
            if len(phrase) >= 3:
                yield phrase


def build_feedback_terms(
    *,
    query: str,
    seed_docs: Sequence[Dict[str, Any]],
    max_terms: int = 14,
) -> List[str]:
    query_tokens = set(_tokenize_for_terms(query))
    scores: Dict[str, float] = {}
    df: Dict[str, int] = {}

    for doc in seed_docs:
        text = str(doc.get("text", "") or "")
        weight = float(doc.get("weight", 1.0) or 1.0)
        seen_in_doc = set()
        for phrase in _extract_phrases(text):
            if not phrase:
                continue
            words = phrase.split()
            # Avoid terms that are purely repeats of the original query. We want
            # terms that add signal, while still allowing multi-word phrases that
            # contain one original word plus new context.
            if len(words) == 1 and words[0] in query_tokens:
                continue
            if len(words) > 4:
                continue
            # mild preference for clinically informative multi-word phrases
            local = weight * (1.0 + 0.25 * (len(words) - 1))
            scores[phrase] = scores.get(phrase, 0.0) + local
            seen_in_doc.add(phrase)
        for phrase in seen_in_doc:
            df[phrase] = df.get(phrase, 0) + 1

    ranked = sorted(
        scores.items(),
        key=lambda kv: (kv[1] + 0.15 * df.get(kv[0], 0), len(kv[0].split()), len(kv[0])),
        reverse=True,
    )

    selected: List[str] = []
    used_words = set()
    for term, _score in ranked:
        words = set(term.split())
        # Reduce repetitive near-duplicates; allow overlap for multi-word terms.
        if len(term.split()) == 1 and term in used_words:
            continue
        if len(term.split()) > 1 and len(words & used_words) >= len(words):
            continue
        selected.append(term)
        used_words.update(words)
        if len(selected) >= max_terms:
            break

    return selected


def build_hybrid_feedback(
    *,
    query: str,
    embedder: Any,
    metas: Sequence[Dict[str, Any]],
    dense_pairs: Sequence[Tuple[int, float]],
    bm25_pairs: Sequence[Tuple[int, float]],
    kg_results: Sequence[Dict[str, Any]] | None = None,
    pmid_to_meta_indices: Dict[str, List[int]] | None = None,
    feedback_docs_per_channel: int = 6,
    alpha: float = 0.75,
    beta: float = 0.25,
    max_doc_chars: int = 1200,
    max_terms: int = 14,
    debug: bool = False,
) -> Tuple[Optional[np.ndarray], str, Dict[str, Any]]:
    """
    Build a hybrid feedback vector and evidence-derived query text.

    The returned vector is intended for second-stage FAISS. The returned query
    text is intended for second-stage BM25 and optional second-stage KG.
    """
    info = HybridFeedbackInfo(
        enabled=True,
        used=False,
        reason="not_started",
        alpha=float(alpha),
        beta=float(beta),
    )

    q = (query or "").strip()
    if not q:
        info.reason = "empty_query"
        return None, "", info.as_dict()

    try:
        feedback_docs_per_channel = max(1, int(feedback_docs_per_channel))
        alpha = float(alpha)
        beta = float(beta)
        if alpha < 0 or beta < 0 or (alpha + beta) <= 0:
            info.reason = "invalid_alpha_beta"
            return None, "", info.as_dict()

        seed_docs: List[Dict[str, Any]] = []
        seen_keys = set()

        dense_norm = _normalize_scores(dense_pairs, higher_is_better=True)
        for idx, _score in list(dense_pairs)[: max(feedback_docs_per_channel * 3, feedback_docs_per_channel)]:
            i = _valid_meta_idx(idx, metas)
            if i is None:
                continue
            key = f"meta::{i}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            _add_seed_doc(
                seed_docs,
                source="dense",
                text=_meta_text(metas[i], max_doc_chars),
                weight=0.70 + 0.30 * float(dense_norm.get(i, 0.0)),
                key=key,
            )
            info.dense_seed_docs += 1
            if info.dense_seed_docs >= feedback_docs_per_channel:
                break

        bm25_norm = _normalize_scores(bm25_pairs, higher_is_better=True)
        for idx, _score in list(bm25_pairs)[: max(feedback_docs_per_channel * 3, feedback_docs_per_channel)]:
            i = _valid_meta_idx(idx, metas)
            if i is None:
                continue
            key = f"meta::{i}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            _add_seed_doc(
                seed_docs,
                source="bm25",
                text=_meta_text(metas[i], max_doc_chars),
                weight=0.70 + 0.30 * float(bm25_norm.get(i, 0.0)),
                key=key,
            )
            info.bm25_seed_docs += 1
            if info.bm25_seed_docs >= feedback_docs_per_channel:
                break

        kg_results = list(kg_results or [])
        pmid_to_meta_indices = pmid_to_meta_indices or {}
        kg_scores = [float(g.get("graph_score", 0.0) or 0.0) for g in kg_results]
        if kg_scores:
            mn, mx = min(kg_scores), max(kg_scores)
        else:
            mn = mx = 0.0

        for g in kg_results[: max(feedback_docs_per_channel * 3, feedback_docs_per_channel)]:
            pmid = g.get("pmid")
            meta = _get_meta_for_pmid(pmid, pmid_to_meta_indices, metas)
            title = str(g.get("title", "") or "")
            disease = str(g.get("disease", "") or "")
            graph_text_parts = [title, disease]
            if meta is not None:
                graph_text_parts.append(_meta_text(meta, max_doc_chars))
            graph_text = "\n".join(p for p in graph_text_parts if p).strip()
            if not graph_text:
                continue
            key = f"kg::{pmid or title}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            raw = float(g.get("graph_score", 0.0) or 0.0)
            norm = 1.0 if mx - mn < 1e-9 else (raw - mn) / (mx - mn)
            _add_seed_doc(
                seed_docs,
                source="kg",
                text=graph_text[:max_doc_chars],
                weight=0.75 + 0.35 * norm,
                key=key,
            )
            info.kg_seed_docs += 1
            if info.kg_seed_docs >= feedback_docs_per_channel:
                break

        if not seed_docs:
            info.reason = "no_seed_docs"
            return None, "", info.as_dict()

        q_vec = embedder.encode([q], normalize_embeddings=True).astype("float32")
        q_vec = l2_normalize(q_vec)

        texts = [d["text"] for d in seed_docs]
        weights = np.asarray([float(d.get("weight", 1.0) or 1.0) for d in seed_docs], dtype="float32")
        weights = np.maximum(weights, 1e-6)
        doc_vecs = embedder.encode(texts, normalize_embeddings=True).astype("float32")
        doc_vecs = l2_normalize(doc_vecs)

        weighted = doc_vecs * weights.reshape(-1, 1)
        centroid = np.sum(weighted, axis=0, keepdims=True) / np.sum(weights)
        centroid = l2_normalize(centroid)

        refined_vec = (alpha * q_vec) + (beta * centroid)
        refined_vec = l2_normalize(refined_vec)

        terms = build_feedback_terms(query=q, seed_docs=seed_docs, max_terms=max_terms)
        feedback_query = (q + " " + " ".join(terms)).strip() if terms else q

        info.used = True
        info.reason = "ok"
        info.feedback_docs_used = len(seed_docs)
        info.feedback_terms_used = len(terms)

        if debug:
            print("Hybrid feedback terms =", terms)
            print("Hybrid feedback info =", info.as_dict())

        return refined_vec, feedback_query, info.as_dict()

    except Exception as exc:
        info.reason = f"error: {type(exc).__name__}: {exc}"
        if debug:
            print("Hybrid feedback failed =", info.as_dict())
        return None, "", info.as_dict()
