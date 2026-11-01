# -*- coding: utf-8 -*-
"""
MedRAG v7.12.6 KG-Guided Agent Tools - Safety Calibration

Tools wrap retrieval, reranking, grounding, candidate generation, evidence
judging, and safety. They are intentionally not diagnostic case-text rules.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import re

from app.rag.utils_text import safe_truncate
from app.rag.evidence_judge import (
    candidate_from_sources,
    judge_candidates,
    select_best_supported_candidate,
    normalize_candidate,
)
from app.rag.common_utils import _safe_float


def _dedup_docs(docs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate while preserving provenance from every retrieval view."""
    out: List[Dict[str, Any]] = []
    positions: Dict[str, int] = {}
    for d in docs or []:
        if not isinstance(d, dict):
            continue
        key = str(d.get("chunk_id") or d.get("pmid") or d.get("title") or id(d))
        if key not in positions:
            item = dict(d)
            item["retrieval_views"] = list(dict.fromkeys(item.get("retrieval_views") or []))
            positions[key] = len(out)
            out.append(item)
            continue

        current = out[positions[key]]
        views = list(current.get("retrieval_views") or []) + list(d.get("retrieval_views") or [])
        channel = d.get("retrieval_channel")
        if channel:
            views.append(str(channel))
        current["retrieval_views"] = list(dict.fromkeys(views))
        for flag in ("from_dense", "from_bm25", "from_kg"):
            current[flag] = bool(current.get(flag) or d.get(flag))
        for score_key in ("dense_score", "bm25_score", "hybrid_score", "score", "medcpt_score", "graph_score", "_grounding_strength"):
            if d.get(score_key) is None:
                continue
            if current.get(score_key) is None or _safe_float(d.get(score_key), float("-inf")) > _safe_float(current.get(score_key), float("-inf")):
                current[score_key] = d.get(score_key)
        for list_key in ("kg_paths", "graph_matched_symptoms", "graph_stroke_families", "graph_territories", "graph_imaging_findings"):
            merged = list(current.get(list_key) or []) + list(d.get(list_key) or [])
            # dicts are deduplicated by repr to remain JSON-safe.
            seen = set(); unique = []
            for value in merged:
                marker = repr(value)
                if marker not in seen:
                    seen.add(marker); unique.append(value)
            if unique:
                current[list_key] = unique
    return out


def _normalize_source(d: Dict[str, Any]) -> Dict[str, Any]:
    source = dict(d)
    source.setdefault("title", d.get("title", ""))
    source.setdefault("text", d.get("text") or d.get("snippet") or "")
    source.setdefault("snippet", d.get("snippet") or safe_truncate(source.get("text", ""), 350))
    source.setdefault("hybrid_score", d.get("hybrid_score", d.get("score", 0.0)))
    source.setdefault("dense_score", d.get("dense_score"))
    source.setdefault("bm25_score", d.get("bm25_score"))
    source.setdefault("from_kg", bool(d.get("from_kg") or d.get("_from_graph") or d.get("graph_score") is not None))
    source.setdefault("from_dense", bool(d.get("from_dense", False)))
    source.setdefault("from_bm25", bool(d.get("from_bm25", False)))
    source.setdefault("retrieval_channel", "kg" if source.get("from_kg") else d.get("retrieval_channel", "hybrid"))
    source.setdefault("retrieval_views", list(d.get("retrieval_views") or [source.get("retrieval_channel")]))
    source.setdefault("evidence_id", str(d.get("chunk_id") or d.get("pmid") or d.get("doi") or d.get("title") or "unknown"))
    source.setdefault("document_id", str(d.get("pmid") or d.get("doi") or d.get("source") or "unknown"))
    source.setdefault("kg_paths", list(d.get("kg_paths") or []))
    return source


def dense_retrieve_tool(query: str, top_k: int = 30) -> List[Dict[str, Any]]:
    """Dense retrieval view. Uses the existing hybrid_search backend and marks channel metadata."""
    from app.rag.retriever import hybrid_search
    docs = hybrid_search(query, top_k=top_k, debug=False)
    out = []
    for d in docs:
        nd = _normalize_source(d)
        nd["from_dense"] = True
        nd["retrieval_channel"] = "hybrid"
        nd["retrieval_views"] = list(dict.fromkeys(list(nd.get("retrieval_views") or []) + ["dense_view"]))
        nd["retrieval_backend_note"] = "shared_hybrid_search_backend"
        out.append(nd)
    return out


def bm25_retrieve_tool(query: str, top_k: int = 30) -> List[Dict[str, Any]]:
    """Lexical retrieval view. Falls back to hybrid_search if internal BM25 is unavailable."""
    # The current project exposes BM25 inside retriever as internal functions.
    # To avoid breaking compatibility, we reuse hybrid_search and annotate the view.
    from app.rag.retriever import hybrid_search
    docs = hybrid_search(query, top_k=top_k, debug=False)
    out = []
    for d in docs:
        nd = _normalize_source(d)
        nd["from_bm25"] = True
        nd["retrieval_channel"] = "hybrid"
        nd["retrieval_views"] = list(dict.fromkeys(list(nd.get("retrieval_views") or []) + ["bm25_view"]))
        nd["retrieval_backend_note"] = "shared_hybrid_search_backend"
        out.append(nd)
    return out


def kg_retrieve_tool(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """KG retrieval view. Reuses hybrid_search and extracts KG-marked results."""
    from app.rag.retriever import hybrid_search
    docs = hybrid_search(query, top_k=max(top_k, 10), debug=False)
    kg_docs = []
    for d in docs:
        nd = _normalize_source(d)
        if nd.get("from_kg") or nd.get("graph_score") is not None:
            nd["from_kg"] = True
            nd["retrieval_channel"] = "kg"
            kg_docs.append(nd)
    return kg_docs[:top_k]


def hybrid_feedback_tool(query: str, seed_docs: Sequence[Dict[str, Any]], max_terms: int = 12) -> Tuple[str, Dict[str, Any]]:
    """Build an evidence-derived feedback query from retrieved seed documents."""
    try:
        from app.rag.hybrid_feedback import build_feedback_terms
        feedback_terms = build_feedback_terms(query=query, seed_docs=list(seed_docs), max_terms=max_terms)
        feedback_query = (query + " " + " ".join(feedback_terms)).strip() if feedback_terms else query
        return feedback_query, {
            "enabled": True,
            "used": bool(feedback_terms),
            "feedback_terms": feedback_terms,
            "feedback_terms_used": len(feedback_terms),
            "reason": "ok" if feedback_terms else "no_terms",
        }
    except Exception as exc:
        return query, {"enabled": True, "used": False, "feedback_terms": [], "reason": f"error: {exc}"}


def second_stage_retrieve_tool(feedback_query: str, top_k: int = 50) -> List[Dict[str, Any]]:
    from app.rag.retriever import hybrid_search
    return [_normalize_source(d) for d in hybrid_search(feedback_query, top_k=top_k, debug=False)]


def rerank_tool(query: str, docs: Sequence[Dict[str, Any]], limit: int = 30) -> List[Dict[str, Any]]:
    docs = _dedup_docs([_normalize_source(d) for d in docs])
    if not docs:
        return []
    try:
        from app.models.medcpt_reranker import rerank_docs
        reranked = rerank_docs(query, docs, text_key="text")
    except Exception:
        reranked = sorted(
            docs,
            key=lambda d: (
                _safe_float(d.get("graph_score"), 0.0),
                _safe_float(d.get("hybrid_score"), _safe_float(d.get("score"), 0.0)),
                _safe_float(d.get("match_cosine"), 0.0),
            ),
            reverse=True,
        )
    return [_normalize_source(d) for d in reranked[:limit]]


def _source_grounding_strength(source: Dict[str, Any]) -> float:
    text = str(source.get("text") or source.get("snippet") or "")
    hybrid = _safe_float(source.get("hybrid_score"), _safe_float(source.get("score"), 0.0))
    medcpt = source.get("medcpt_score")
    cosine = _safe_float(source.get("match_cosine"), 0.0)
    graph = _safe_float(source.get("graph_score"), 0.0)
    strength = 0.0
    if len(text.strip()) >= 120:
        strength += 0.20
    elif len(text.strip()) >= 60:
        strength += 0.10
    if hybrid >= 0.70:
        strength += 0.30
    elif hybrid >= 0.50:
        strength += 0.22
    elif hybrid >= 0.35:
        strength += 0.14
    elif hybrid >= 0.20:
        strength += 0.08
    if medcpt is not None:
        m = _safe_float(medcpt, 0.0)
        if m >= 0.80:
            strength += 0.25
        elif m >= 0.60:
            strength += 0.18
        elif m >= 0.40:
            strength += 0.10
        elif m >= 5.0:  # KG-protected MedCPT-style floors in older pipeline
            strength += 0.25
    if cosine >= 0.75:
        strength += 0.25
    elif cosine >= 0.60:
        strength += 0.18
    elif cosine >= 0.45:
        strength += 0.10
    if graph >= 0.70:
        strength += 0.18
    elif graph >= 0.45:
        strength += 0.10
    return round(min(strength, 1.0), 4)


def grounding_tool(case_text: str, docs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not docs:
        return {"allow_llm": False, "reason": "no_sources", "top_strength": 0.0, "support_count": 0, "avg_strength_top3": 0.0}
    strengths = []
    for d in docs:
        s = _source_grounding_strength(d)
        d["_grounding_strength"] = s
        strengths.append(s)
    strengths_sorted = sorted(strengths, reverse=True)
    top = strengths_sorted[0] if strengths_sorted else 0.0
    avg3 = sum(strengths_sorted[:3]) / max(min(len(strengths_sorted), 3), 1)
    support_count = sum(1 for x in strengths_sorted if x >= 0.45)
    allow = top >= 0.55 and (support_count >= 1 or avg3 >= 0.45)
    return {
        "allow_llm": bool(allow),
        "reason": "grounded" if allow else "weak_grounding",
        "top_strength": round(top, 4),
        "support_count": int(support_count),
        "avg_strength_top3": round(avg3, 4),
    }


# ============================================================
# Candidate KG verification helpers
# ============================================================

ISCHEMIC_CANDIDATE_KEYS = {
    "acute_ischemic_stroke",
    "large_vessel_occlusion",
    "anterior_circulation_stroke",
    "posterior_circulation_stroke",
    "brainstem_infarction",
    "cerebellar_infarction",
    "lacunar_infarction",
    "basilar_artery_occlusion",
    "vertebrobasilar_insufficiency",
}

HEMORRHAGE_CANDIDATE_KEYS = {
    "intracerebral_hemorrhage",
    "subarachnoid_hemorrhage",
    "hemorrhagic_stroke",
}

TIA_CANDIDATE_KEYS = {"tia"}


def _low_text(x: Any) -> str:
    return str(x or "").lower()


def _source_family_text(source: Dict[str, Any]) -> str:
    return " ".join(
        str(source.get(k) or "")
        for k in (
            "title", "text", "snippet", "graph_disease", "disease",
            "graph_category", "category", "graph_article_type", "article_type",
            "candidate_key", "candidate_label",
        )
    ).lower()


def _is_hemorrhagic_doc(source: Dict[str, Any]) -> bool:
    txt = _source_family_text(source)
    return any(
        term in txt
        for term in [
            "hemorrhage", "haemorrhage", "hemorrhagic", "haemorrhagic",
            "intracerebral hemorrhage", "subarachnoid hemorrhage",
            "intraparenchymal hematoma", "parenchymal hematoma", "hematoma",
            "sah", "ich",
        ]
    ) or _low_text(source.get("graph_category") or source.get("category")) == "hemorrhagic"


def _is_ischemic_doc(source: Dict[str, Any]) -> bool:
    txt = _source_family_text(source)
    if _is_hemorrhagic_doc(source):
        return False
    return any(
        term in txt
        for term in [
            "ischemic stroke", "ischaemic stroke", "acute ischemic stroke",
            "acute ischaemic stroke", "cerebral infarction", "brain infarction",
            "infarction", "infarct", "arterial ischemic", "arterial ischaemic",
            "large vessel occlusion", "middle cerebral artery", "mca",
            "thrombectomy", "thrombolysis", "alteplase", "tpa",
            "posterior circulation", "vertebrobasilar", "brainstem", "cerebellar",
        ]
    ) or _low_text(source.get("graph_category") or source.get("category")) in {"vascular", "posterior_vascular"}


def _has_strong_infarct_confirmation(source: Dict[str, Any]) -> bool:
    txt = _source_family_text(source)
    return any(
        term in txt
        for term in [
            "dwi lesion", "diffusion restriction", "restricted diffusion",
            "mri confirmed infarct", "mri-confirmed infarct",
            "ct confirmed acute infarct", "ct-confirmed acute infarct",
            "imaging-confirmed infarction", "imaging confirmed infarction",
            "infarct on mri", "infarct on ct",
        ]
    )



NOISY_KG_DOC_TERMS = [
    "benign positioning vertigo",
    "benign paroxysmal positional vertigo",
    "bppv",
    "vestibular rehabilitation",
    "vestibular hypofunction",
    "olivo-ponto-cerebellar atrophy",
    "olivopontocerebellar atrophy",
    "spinocerebellar ataxia",
    "ataxia telangiectasia",
    "captive african lion cub",
    "arnold-chiari malformation",
    "masseter reflex potentials",
    "animal model",
    "mouse",
    "mice",
    "rat",
    "rats",
    "murine",
    "in vitro",
    "cell culture",
    "protein kinase",
    "phosphorylation",
    "dna repair",
    "gamma-h2ax",
]

NOISY_KG_RESCUE_TERMS = [
    "acute ischemic stroke",
    "acute ischaemic stroke",
    "posterior circulation stroke",
    "brainstem infarction",
    "cerebellar infarction",
    "midbrain infarction",
    "pontine infarction",
    "medullary infarction",
    "intracerebral hemorrhage",
    "intracerebral haemorrhage",
    "subarachnoid hemorrhage",
    "subarachnoid haemorrhage",
    "transient ischemic attack",
    "transient ischaemic attack",
]


# [تم حذف كود ميت مكرر غير مستخدم: _is_noisy_kg_doc (old) — السطر الأصلي 371-396]
# [تم حذف كود ميت مكرر غير مستخدم: _candidate_family_allows_doc (old) — السطر الأصلي 397-450]
def _candidate_queries(candidate_label: str, aliases: Sequence[str], case_text: str, intent: Optional[Dict[str, Any]]) -> List[str]:
    """Build KG queries. These are retrieval queries, not diagnostic rules."""
    intent = intent or {}
    base = (case_text or "").strip()
    label = candidate_label.strip()
    alias_terms = " ".join(str(a) for a in list(aliases or [])[:5])

    hints: List[str] = []
    if intent.get("acute_neuro"):
        hints.append("acute stroke emergency")
    if intent.get("focal_neuro"):
        hints.append("focal neurological deficit")
    if intent.get("hemorrhage_warning"):
        hints.append("hemorrhage hematoma subarachnoid intracerebral")
    if intent.get("transient_resolved_episode"):
        hints.append("transient ischemic attack resolved symptoms")

    query_variants = [
        f"{base} {label}".strip(),
        f"{base} {alias_terms}".strip(),
        f"{base} {label} {' '.join(hints)}".strip(),
    ]

    seen = set()
    out = []
    for q in query_variants:
        q = re.sub(r"\s+", " ", q).strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def candidate_kg_verification_tool(
    case_text: str,
    candidates: Optional[Sequence[str]] = None,
    intent: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]] | Dict[str, Any]:
    """
    Verify each candidate against KG evidence.

    The KG is used as structured evidence only. This function does not diagnose
    from case_text; it asks the graph for candidate-related evidence and then
    filters cross-family evidence.
    """
    intent = intent or {}
    legacy_single_candidate_call = candidates is None
    if candidates is None:
        # Backward-compatible API used by older tests:
        # candidate_kg_verification_tool("Transient ischemic attack / TIA", top_k=2)
        # In that form the first argument is the candidate label, not the case text.
        candidate_label_for_legacy = str(case_text or "")
        case_text = candidate_label_for_legacy
        candidates = [candidate_label_for_legacy]

    verifications: Dict[str, Any] = {}
    all_docs: List[Dict[str, Any]] = []

    try:
        import yaml
        from app.knowledge_graph.graph_retriever import GraphRetriever

        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        kg_cfg = cfg.get("knowledge_graph", {}) or {}
        retriever = GraphRetriever(
            uri=kg_cfg.get("uri", "bolt://127.0.0.1:7687"),
            user=kg_cfg.get("user", "neo4j"),
            password=kg_cfg.get("password", "password"),
        )
    except Exception as exc:
        for candidate in candidates or []:
            verifications[str(candidate)] = {
                "candidate": str(candidate),
                "candidate_label": str(candidate),
                "candidate_key": None,
                "kg_query_used": "",
                "kg_support_score": 0.0,
                "source_count": 0,
                "docs": [],
                "error": f"kg_unavailable: {exc}",
            }
        if legacy_single_candidate_call:
            first_candidate = str(list(candidates or [""])[0])
            return verifications.get(first_candidate, {
                "candidate": first_candidate,
                "candidate_label": first_candidate,
                "candidate_key": None,
                "kg_support_score": 0.0,
                "source_count": 0,
                "docs": [],
                "error": "kg_unavailable",
            })
        return verifications, []

    for candidate in list(dict.fromkeys(candidates or [])):
        norm = normalize_candidate(str(candidate))
        if norm:
            candidate_key, candidate_label, aliases, categories = norm
        else:
            candidate_key, candidate_label, aliases, categories = "unknown", str(candidate), [], []

        docs: List[Dict[str, Any]] = []
        queries = _candidate_queries(candidate_label, aliases, case_text, intent)

        for q in queries:
            try:
                if hasattr(retriever, "search_by_candidate"):
                    found = retriever.search_by_candidate(
                        candidate_label=candidate_label,
                        candidate_key=candidate_key,
                        aliases=aliases,
                        case_text=case_text,
                        clinical_intent=intent,
                        top_k=max(top_k, 5),
                        candidate_query=q,
                    )
                else:
                    found = retriever.search(q, top_k=max(top_k, 5))
            except Exception:
                found = []

            for d in found or []:
                if not isinstance(d, dict):
                    continue
                nd = _normalize_source(d)
                nd["from_kg"] = True
                nd["retrieval_channel"] = "candidate_kg"
                nd.setdefault("source", "GraphKB")
                nd["candidate_key"] = candidate_key
                nd["candidate_label"] = candidate_label
                nd["kg_query_used"] = q
                if nd.get("disease") and not nd.get("graph_disease"):
                    nd["graph_disease"] = nd.get("disease")
                if nd.get("category") and not nd.get("graph_category"):
                    nd["graph_category"] = nd.get("category")
                if nd.get("article_type") and not nd.get("graph_article_type"):
                    nd["graph_article_type"] = nd.get("article_type")
                if not _candidate_family_allows_doc(candidate_key, nd, intent):
                    continue
                docs.append(nd)

        docs = _dedup_docs(docs)
        docs.sort(key=lambda d: (_safe_float(d.get("graph_score"), 0.0), _safe_float(d.get("hybrid_score"), 0.0)), reverse=True)
        docs = docs[:top_k]
        kg_score = max([_safe_float(d.get("graph_score"), 0.0) for d in docs] or [0.0])

        record = {
            "candidate": str(candidate),
            "candidate_label": candidate_label,
            "candidate_key": candidate_key,
            "aliases_used": list(aliases or [])[:8],
            "kg_queries_used": queries,
            "kg_query_used": queries[0] if queries else "",
            "kg_support_score": round(float(kg_score), 4),
            "source_count": len(docs),
            "docs": docs[:5],
            "kg_paths": [path for doc in docs for path in list(doc.get("kg_paths") or [])][:20],
        }
        verifications[str(candidate)] = record
        verifications[candidate_label] = record
        verifications[candidate_key] = record
        all_docs.extend(docs)

    if legacy_single_candidate_call:
        first_candidate = str(list(candidates or [""])[0])
        norm = normalize_candidate(first_candidate)
        possible_keys = [first_candidate]
        if norm:
            possible_keys.extend([norm[1], norm[0]])
        for k in possible_keys:
            if k in verifications:
                return verifications[k]
        return {
            "candidate": first_candidate,
            "candidate_label": first_candidate,
            "candidate_key": norm[0] if norm else None,
            "kg_support_score": 0.0,
            "source_count": 0,
            "docs": [],
        }

    return verifications, _dedup_docs(all_docs)

def candidate_generation_tool(
    case_text: str,
    docs: Sequence[Dict[str, Any]],
    match_top: Optional[Dict[str, Any]] = None,
    initial_answer: str | None = None,
    intent: Optional[Dict[str, Any]] = None,
) -> List[str]:
    evidence_docs = list(docs or [])
    if match_top:
        evidence_docs = [match_top] + evidence_docs
    return candidate_from_sources(evidence_docs, initial_answer=initial_answer, max_candidates=8, intent=intent)


def evidence_judge_tool(
    case_text: str,
    candidates: Sequence[str],
    docs: Sequence[Dict[str, Any]],
    grounding: Dict[str, Any],
    *,
    evidence_support_cap: int = 3,
    intent: Optional[Dict[str, Any]] = None,
    candidate_kg_verifications: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    return judge_candidates(
        case_text,
        candidates,
        docs,
        grounding,
        evidence_support_cap=evidence_support_cap,
        intent=intent,
        candidate_kg_verifications=candidate_kg_verifications,
    )


def final_answer_tool(best_candidate: Optional[Dict[str, Any]], evidence_judgments: Sequence[Dict[str, Any]]) -> str:
    if not best_candidate:
        return "Evidence is insufficient"
    return str(best_candidate.get("candidate") or "Evidence is insufficient").strip() or "Evidence is insufficient"


# [تم حذف كود ميت مكرر غير مستخدم: safety_guard_tool (old) — السطر الأصلي 680-715]
def preserve_kg_sources_tool(
    reranked_docs: Sequence[Dict[str, Any]],
    kg_docs: Sequence[Dict[str, Any]],
    min_kg: int = 3,
    max_total: int = 30,
) -> List[Dict[str, Any]]:
    """
    Preserve candidate/KG evidence after reranking.

    Backward-compatible tool used by existing tests and by older controller paths.
    It keeps the strongest KG docs in the final evidence set so KG contribution
    does not disappear after dense/BM25/MedCPT ranking.

    This function does not diagnose from case text. It only merges already
    retrieved evidence sources and marks KG metadata clearly.
    """
    max_total = max(1, int(max_total or 30))
    min_kg = max(0, int(min_kg or 0))

    normalized_reranked = [_normalize_source(d) for d in (reranked_docs or []) if isinstance(d, dict)]
    normalized_kg = [_normalize_source(d) for d in (kg_docs or []) if isinstance(d, dict)]

    kg_ranked = sorted(
        normalized_kg,
        key=lambda d: (
            _safe_float(d.get("graph_score"), 0.0),
            _safe_float(d.get("_grounding_strength"), 0.0),
            _safe_float(d.get("medcpt_score"), 0.0),
            _safe_float(d.get("hybrid_score", d.get("score", 0.0)), 0.0),
        ),
        reverse=True,
    )

    preserved: List[Dict[str, Any]] = []
    for d in kg_ranked:
        if len(preserved) >= min_kg:
            break
        dd = dict(d)
        dd["from_kg"] = True
        dd.setdefault("retrieval_channel", "candidate_kg")
        dd["kg_preserved"] = True
        preserved.append(dd)

    merged = _dedup_docs(preserved + normalized_reranked + normalized_kg)

    # If final list is longer than max_total, keep preserved KG docs first and fill
    # remaining slots from the highest-ranked non-duplicate reranked evidence.
    if len(merged) > max_total:
        final: List[Dict[str, Any]] = []
        seen = set()

        def _doc_key(d: Dict[str, Any]) -> str:
            return str(d.get("chunk_id") or d.get("pmid") or d.get("title") or id(d))

        for d in preserved:
            k = _doc_key(d)
            if k not in seen and len(final) < max_total:
                seen.add(k)
                final.append(d)

        for d in normalized_reranked + kg_ranked:
            k = _doc_key(d)
            if k not in seen and len(final) < max_total:
                seen.add(k)
                final.append(d)

        info = {
            "enabled": True,
            "min_kg": min_kg,
            "preserved": len(preserved),
            "final_kg_count": sum(1 for d in final if d.get("from_kg") or d.get("retrieval_channel") == "candidate_kg"),
            "final_total": len(final),
        }
        return final, info

    final = merged[:max_total]
    info = {
        "enabled": True,
        "min_kg": min_kg,
        "preserved": len(preserved),
        "final_kg_count": sum(1 for d in final if d.get("from_kg") or d.get("retrieval_channel") == "candidate_kg"),
        "final_total": len(final),
    }
    return final, info


# ============================================================
# v7.15 candidate-KG quality and subtype alignment gates
# ============================================================
from app.rag.clinical_context import analyze_case_context as _v715_case_profile

_V715_NONHUMAN_OR_PRECLINICAL = [
    "animal model", "experimental model", "in vitro", "cell culture",
    "mouse", "mice", "murine", "rat", "rats", "rabbit", "rabbits",
    "dog", "dogs", "canine", "cat", "cats", "feline", "pig", "pigs",
    "porcine", "swine", "sheep", "goat", "monkey", "macaque", "primate",
    "zebrafish", "guinea pig", "protein kinase", "phosphorylation",
    "gene expression", "dna repair", "gamma-h2ax",
]


def _v715_contains_word(text: str, term: str) -> bool:
    low = _low_text(text)
    t = _low_text(term).strip()
    if len(t) <= 4 and re.fullmatch(r"[a-z0-9]+", t):
        return re.search(rf"\b{re.escape(t)}\b", low) is not None
    return t in low


def _is_noisy_kg_doc(source: Dict[str, Any]) -> bool:
    txt = _source_family_text(source)
    title = _low_text(source.get("title") or "")
    if any(_v715_contains_word(title, term) or _v715_contains_word(txt, term) for term in _V715_NONHUMAN_OR_PRECLINICAL):
        return True
    # Preserve the legacy chronic/mimic filters as an additional gate.
    legacy_noise = any(t in txt for t in NOISY_KG_DOC_TERMS)
    if not legacy_noise:
        return False
    title_noise = any(t in title for t in NOISY_KG_DOC_TERMS)
    title_rescue = any(t in title for t in NOISY_KG_RESCUE_TERMS)
    return bool(title_noise and not title_rescue)


def _candidate_family_allows_doc(candidate_key: str, source: Dict[str, Any], intent: Optional[Dict[str, Any]]) -> bool:
    intent = intent or {}
    txt = _source_family_text(source)
    title = _low_text(source.get("title") or "")
    disease = _low_text(source.get("graph_disease") or source.get("disease") or "")
    primary = f"{title} {disease}".strip()
    graph_score = _safe_float(source.get("graph_score"), 0.0)
    alignment = _safe_float(source.get("candidate_alignment_score"), 0.0)

    if _is_noisy_kg_doc(source):
        return False

    context = intent.get("context_profile") if isinstance(intent.get("context_profile"), dict) else {}
    if not context and intent.get("case_text"):
        context = _v715_case_profile(str(intent.get("case_text")))

    # Metadata-only KG rows need an explicit candidate-family match.  A generic
    # category plus high graph score is not enough to support a subtype.
    has_textual_content = bool(str(source.get("text") or source.get("snippet") or "").strip())
    has_paths = bool(source.get("kg_paths"))
    if not has_textual_content and not has_paths:
        return False

    exact_terms: Dict[str, List[str]] = {
        "subarachnoid_hemorrhage": [
            "subarachnoid hemorrhage", "subarachnoid haemorrhage",
            "aneurysmal subarachnoid", "ruptured aneurysm", "basal cistern blood",
        ],
        "intracerebral_hemorrhage": [
            "intracerebral hemorrhage", "intracerebral haemorrhage",
            "intraparenchymal hemorrhage", "intraparenchymal haemorrhage",
            "intraparenchymal hematoma", "parenchymal hematoma", "basal ganglia hematoma",
        ],
        "hemorrhagic_stroke": [
            "hemorrhagic stroke", "haemorrhagic stroke", "intracranial hemorrhage",
            "intracranial haemorrhage", "brain hemorrhage", "brain haemorrhage",
        ],
        "tia": ["transient ischemic attack", "transient ischaemic attack", "amaurosis fugax"],
        "acute_ischemic_stroke": [
            "acute ischemic stroke", "acute ischaemic stroke", "ischemic stroke",
            "ischaemic stroke", "cerebral infarction", "brain infarction",
        ],
        "large_vessel_occlusion": ["large vessel occlusion", "mca occlusion", "carotid terminus", "thrombectomy"],
        "anterior_circulation_stroke": ["anterior circulation", "mca stroke", "middle cerebral artery stroke"],
        "posterior_circulation_stroke": ["posterior circulation stroke", "vertebrobasilar stroke"],
        "brainstem_infarction": ["brainstem infarction", "pontine infarction", "medullary infarction"],
        "cerebellar_infarction": ["cerebellar infarction", "cerebellar stroke"],
        "basilar_artery_occlusion": ["basilar artery occlusion", "basilar occlusion"],
        "lacunar_infarction": ["lacunar infarction", "lacunar stroke", "small vessel ischemic stroke"],
    }
    exact_match = any(_v715_contains_word(primary, term) for term in exact_terms.get(candidate_key, []))

    # Do not let generic hemorrhage KG evidence support a hemorrhage subtype in
    # a case without affirmed hemorrhage findings.  It may still remain in the
    # global retrieval pool as a differential/conflicting source.
    if candidate_key == "subarachnoid_hemorrhage":
        if context and not context.get("sah_warning"):
            return False
        return bool(exact_match and (graph_score >= 0.25 or alignment >= 0.12))

    if candidate_key == "intracerebral_hemorrhage":
        if context and not context.get("ich_warning"):
            return False
        return bool(exact_match and (graph_score >= 0.25 or alignment >= 0.12))

    if candidate_key == "hemorrhagic_stroke":
        if context and not context.get("hemorrhage_warning"):
            return False
        return bool(exact_match and (graph_score >= 0.25 or alignment >= 0.12))

    if candidate_key in TIA_CANDIDATE_KEYS:
        if intent.get("hemorrhage_warning") or intent.get("persistent_deficit"):
            return False
        if not intent.get("transient_resolved_episode"):
            return False
        if _has_strong_infarct_confirmation(source):
            return False
        return bool(exact_match and (graph_score >= 0.25 or alignment >= 0.12))

    if candidate_key in ISCHEMIC_CANDIDATE_KEYS:
        if intent.get("hemorrhage_warning") and not _has_strong_infarct_confirmation(source):
            return False
        if _is_hemorrhagic_doc(source) and not _has_strong_infarct_confirmation(source):
            return False
        if intent.get("transient_resolved_episode") and not intent.get("persistent_deficit") and not _has_strong_infarct_confirmation(source):
            return False
        if graph_score < 0.20:
            return False
        # Require exact subtype alignment when a subtype is requested.  The
        # generic AIS parent may accept an ischemic-family match.
        if candidate_key == "acute_ischemic_stroke":
            return bool(exact_match or (_is_ischemic_doc(source) and alignment >= 0.12))
        return bool(exact_match and alignment >= 0.08)

    return bool(exact_match or alignment >= 0.15)

# Negation-aware safety guard.  A denied seizure/hypoglycemia must not block a
# focal stroke presentation as a mimic.
from app.rag.clinical_context import (
    analyze_case_context as _v715_guard_profile,
    affirmed_terms as _v715_affirmed_terms,
)


def safety_guard_tool(case_text: str, answer: str, grounding: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    text = str(case_text or "")
    profile = _v715_guard_profile(text)
    flags: Dict[str, Any] = {
        "blocked": False,
        "reasons": [],
        "clinical_context_source": profile.get("source"),
    }

    cardiac = _v715_affirmed_terms(text, [
        "chest pain", "st elevation", "stemi", "myocardial infarction", "acute coronary syndrome"
    ])
    if cardiac and not profile.get("focal_neuro"):
        flags["blocked"] = True
        flags["reasons"].append("out_of_domain_cardiac_without_focal_neuro_deficit")

    metabolic = _v715_affirmed_terms(text, [
        "hypoglycemia", "hypoglycaemia", "low blood glucose", "low glucose", "نقص سكر"
    ])
    if metabolic:
        flags["blocked"] = True
        flags["reasons"].append("affirmed_metabolic_mimic")

    seizure_mimic = _v715_affirmed_terms(text, [
        "generalized seizure", "tonic-clonic seizure", "postictal weakness", "postictal state", "todds paralysis"
    ])
    if seizure_mimic and not profile.get("persistent_deficit"):
        flags["blocked"] = True
        flags["reasons"].append("affirmed_seizure_mimic_without_persistent_focal_deficit")

    peripheral = _v715_affirmed_terms(text, ["bppv", "positional vertigo", "triggered by turning in bed"])
    if peripheral and not profile.get("focal_neuro"):
        flags["blocked"] = True
        flags["reasons"].append("affirmed_peripheral_vertigo_mimic")

    if not grounding.get("allow_llm") and _safe_float(grounding.get("top_strength"), 0.0) < 0.45:
        flags["blocked"] = True
        flags["reasons"].append("weak_grounding")

    if flags["blocked"]:
        return "Evidence is insufficient", flags
    return answer or "Evidence is insufficient", flags
