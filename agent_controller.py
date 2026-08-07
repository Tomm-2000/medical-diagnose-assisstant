# -*- coding: utf-8 -*-
"""
MedRAG v7.12.3 KG-Guided Evidence-Based RAG Controller

The controller coordinates tools. It does not perform diagnostic if/else rules
from case text. Subtype selection is delegated to evidence_judge.py and requires
support from retrieved evidence, KG metadata, and grounding.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import yaml
from app.rag.common_utils import _load_cfg

from app.rag.agent_state import MedRAGAgentState
from app.rag.clinical_intent import describe_clinical_intent
from app.rag.clinical_signals import detect_clinical_signals
from app.rag.tools import (
    dense_retrieve_tool,
    bm25_retrieve_tool,
    kg_retrieve_tool,
    hybrid_feedback_tool,
    second_stage_retrieve_tool,
    rerank_tool,
    grounding_tool,
    candidate_generation_tool,
    candidate_kg_verification_tool,
    evidence_judge_tool,
    final_answer_tool,
    safety_guard_tool,
    _dedup_docs,
)
from app.rag.evidence_judge import select_best_supported_candidate, explain_support
from app.rag.explainability import build_explanation

PIPELINE_VERSION = "v7.14-decision-trace-sourcecheckup"


def _confidence_from_grounding_and_judgment(grounding: Dict[str, Any], selected: Optional[Dict[str, Any]]) -> str:
    #تتأكد الدالة مما إذا كان هناك تشخيص مرشح (selected)
    if not selected:
        return "منخفض"
    
    #تقرأ قيمة support_score من القاموس selected (وهي درجة دعم التشخيص)
    score = float(selected.get("support_score") or 0.0)
    top = float(grounding.get("top_strength") or 0.0)
    if score >= 0.80 and top >= 0.65:
        return "مرتفع"
    if score >= 0.66 and top >= 0.50:
        return "متوسط"
    return "منخفض"


#تُطبق دالة max باختيار مفتاح مقارنة مبني على Tuple مقوّمة بـ 3 أوزان متتالية مرتبة حسب الأولوية:
def _match_top(docs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not docs:
        return None
    best = max(
        docs,
        key=lambda d: (
            float(d.get("_grounding_strength") or 0.0),
            float(d.get("graph_score") or 0.0),
            float(d.get("hybrid_score") or d.get("score") or 0.0),
        ),
    )
    return {
        "pmid": best.get("pmid"),
        "title": best.get("title"),
        "hybrid_score": best.get("hybrid_score", best.get("score")),
        "medcpt_score": best.get("medcpt_score"),
        "graph_score": best.get("graph_score"),
        "graph_disease": best.get("graph_disease"),
        "graph_category": best.get("graph_category"),
        "from_kg": bool(best.get("from_kg") or best.get("graph_score") is not None),
        "grounding_strength": best.get("_grounding_strength"),
    }


def run_medrag_agent(case_text: str, top_k: int | None = None, debug: bool = False) -> Dict[str, Any]:
    cfg = _load_cfg()
    agent_cfg = cfg.get("agentic_mode", {}) or {}
    evidence_cap = int(agent_cfg.get("evidence_support_cap", 3) or 3)
    min_support = float(agent_cfg.get("min_candidate_support_score", 0.66) or 0.66)

    k = int(top_k or (cfg.get("retrieval", {}) or {}).get("top_k_merged", 30) or 30)
    state = MedRAGAgentState(case_text=case_text, normalized_case=(case_text or "").strip())

    # 1) Lightweight intent: route/safety descriptor only.
    state.intent = describe_clinical_intent(state.normalized_case)
    try:
        state.clinical_signals = detect_clinical_signals(state.normalized_case)
    except Exception:
        state.clinical_signals = detect_clinical_signals(state.normalized_case, use_model=False)
    state.add_step("intent_descriptor_v2", output=state.intent)
    state.add_step("clinical_signal_detection", output=state.clinical_signals)

    # 2) Tool-based retrieval views.
    dense_docs = dense_retrieve_tool(state.normalized_case, top_k=max(10, k))
    bm25_docs = bm25_retrieve_tool(state.normalized_case, top_k=max(10, k))
    kg_docs = kg_retrieve_tool(state.normalized_case, top_k=max(4, min(10, k)))
    state.initial_docs = _dedup_docs(dense_docs + bm25_docs + kg_docs)
    state.add_step("initial_retrieval", dense=len(dense_docs), bm25=len(bm25_docs), kg=len(kg_docs), merged=len(state.initial_docs))

    # 3) Evidence-derived hybrid feedback.
    state.feedback_query, fb_info = hybrid_feedback_tool(state.normalized_case, state.initial_docs)
    state.add_step("hybrid_feedback", **fb_info)

    # 4) Second-stage retrieval.
    if fb_info.get("used") and state.feedback_query and state.feedback_query != state.normalized_case:
        state.second_stage_docs = second_stage_retrieve_tool(state.feedback_query, top_k=max(k, 30))
    else:
        state.second_stage_docs = []
    state.add_step("second_stage_retrieval", docs=len(state.second_stage_docs))

    # 5) Merge and rerank.
    merged = _dedup_docs(state.initial_docs + state.second_stage_docs)
    state.reranked_docs = rerank_tool(state.normalized_case, merged, limit=max(k, 30))
    state.add_step(
        "reranking",
        docs=len(state.reranked_docs),
        top_docs=[{"pmid": d.get("pmid"), "title": d.get("title"), "graph_disease": d.get("graph_disease")} for d in state.reranked_docs[:5]],
    )

    # 6) Grounding.
    state.grounding = grounding_tool(state.normalized_case, state.reranked_docs)
    state.add_step("grounding", **state.grounding)

    # 7) Candidate generation from evidence only.
    match_top = _match_top(state.reranked_docs)
    state.candidate_diagnoses = candidate_generation_tool(state.normalized_case, state.reranked_docs, match_top=match_top, intent=state.intent)
    state.add_step("candidate_generation", candidates=state.candidate_diagnoses)

    # 8) Candidate-specific KG verification.
    kg_top_k = int(agent_cfg.get("candidate_kg_top_k", 5) or 5)
    state.candidate_kg_verifications, state.candidate_kg_docs = candidate_kg_verification_tool(
        state.normalized_case,
        state.candidate_diagnoses,
        intent=state.intent,
        top_k=kg_top_k,
    )
    kg_docs_added = len(state.candidate_kg_docs or [])
    state.add_step(
        "candidate_kg_verification",
        candidates_checked=len(state.candidate_diagnoses or []),
        kg_docs_added=kg_docs_added,
        candidates_with_kg=sum(1 for v in (state.candidate_kg_verifications or {}).values() if isinstance(v, dict) and int(v.get("source_count") or 0) > 0),
    )

    # 9) Preserve candidate-KG evidence and recompute grounding.
    state.final_docs = _dedup_docs(list(state.reranked_docs or []) + list(state.candidate_kg_docs or []))
    if kg_docs_added:
        state.final_docs = rerank_tool(state.normalized_case, state.final_docs, limit=max(k, 30))
        state.add_step(
            "kg_preservation",
            enabled=True,
            preserved=kg_docs_added,
            final_kg_count=sum(1 for d in state.final_docs if d.get("retrieval_channel") == "candidate_kg"),
            final_total=len(state.final_docs),
        )
    else:
        state.add_step("kg_preservation", enabled=True, preserved=0, final_kg_count=0, final_total=len(state.final_docs))

    state.grounding = grounding_tool(state.normalized_case, state.final_docs)
    state.add_step("grounding_after_candidate_kg", **state.grounding)

    # 10) Evidence judge.
    state.evidence_judgments = evidence_judge_tool(
        state.normalized_case,
        state.candidate_diagnoses,
        state.final_docs,
        state.grounding,
        evidence_support_cap=evidence_cap,
        intent=state.intent,
        candidate_kg_verifications=state.candidate_kg_verifications,
    )
    selected = select_best_supported_candidate(state.evidence_judgments, min_support_score=min_support)
    state.selected_candidate = dict(selected or {})
    state.supporting_sources = list(state.selected_candidate.get("supporting_sources") or [])
    state.conflicting_sources = list(state.selected_candidate.get("conflicting_sources") or [])
    state.add_step(
        "evidence_judge",
        selected=selected.get("candidate") if selected else None,
        support_score=selected.get("support_score") if selected else 0.0,
        reason=explain_support(selected) if selected else "no supported candidate",
    )

    # 11) Final answer and safety guard.
    raw_answer = final_answer_tool(selected, state.evidence_judgments)
    final_answer, safety_flags = safety_guard_tool(state.normalized_case, raw_answer, state.grounding)
    state.final_answer = final_answer
    state.safety_flags = safety_flags
    state.add_step("final_safety_guard", passed=not safety_flags.get("blocked", False), flags=safety_flags)

    sources = (state.final_docs or state.reranked_docs)[:k]
    confidence = _confidence_from_grounding_and_judgment(state.grounding, selected)
    if state.final_answer == "Evidence is insufficient":
        confidence = "منخفض"

    out = {
        "answer": state.final_answer,
        "confidence": confidence,
        "top_score": float(sources[0].get("hybrid_score", sources[0].get("score", 0.0)) or 0.0) if sources else 0.0,
        "sources": sources,
        "pipeline_version": PIPELINE_VERSION,
        "used_fallback": state.final_answer == "Evidence is insufficient",
        "match_top": match_top,
        "grounding": state.grounding,
        "clinical_intent": state.intent,
        "clinical_signals": state.clinical_signals,
        "candidate_diagnoses": state.candidate_diagnoses,
        "candidate_kg_verifications": state.candidate_kg_verifications,
        "top_kg_sources": [d for d in (state.candidate_kg_docs or [])[:8]],
        "evidence_judgments": state.evidence_judgments,
        "selected_candidate": selected,
        "supporting_sources": state.supporting_sources,
        "conflicting_sources": state.conflicting_sources,
        "agent_steps": state.agent_steps,
        "safety_flags": state.safety_flags,
        "debug_info": state.debug_info if debug else {},
        "legacy_rules_used": False,
        "legacy_resolver_used": False,
        "agentic_mode": True,
    }

    # v7.12.11 Explainability layer. This is post-decision only: it does not
    # alter final_answer, selected_candidate, support_score, retrieval, KG, or grounding.
    try:
        out["explanation"] = build_explanation(
            case_text=state.normalized_case,
            final_answer=state.final_answer,
            selected_candidate=selected,
            candidates=state.evidence_judgments or state.candidate_diagnoses,
            clinical_signals=state.clinical_signals,
            clinical_intent=state.intent,
            retrieved_sources=sources,
            kg_sources=state.candidate_kg_docs or [],
            grounding=state.grounding,
            evidence_judge_reason=explain_support(selected) if selected else "no supported candidate",
            candidate_kg_verifications=state.candidate_kg_verifications,
            safety_flags=state.safety_flags,
            debug_info={
                "agent_steps": state.agent_steps if debug else [],
                "candidate_diagnoses": state.candidate_diagnoses,
                "candidate_kg_verifications": state.candidate_kg_verifications,
                "safety_flags": state.safety_flags,
            },
        )
        state.explanation = dict(out["explanation"] or {})
        out["explainability_included"] = True
    except Exception as exc:
        out["explainability_included"] = False
        out["explanation"] = {
            "explainability_version": "v7.12.11-explainability",
            "final_answer": state.final_answer,
            "selected_candidate": selected.get("candidate_key") if isinstance(selected, dict) else None,
            "error": f"explainability_failed: {type(exc).__name__}: {exc}",
            "medical_safety_notes": [
                "This is decision-support only and not a final clinical diagnosis.",
                "Neuroimaging and physician assessment are required for acute stroke decisions.",
            ],
        }

    return out



