# -*- coding: utf-8 -*-
"""
MedRAG v7.13 SourceCheckup-inspired Explainability Layer.

The layer is deliberately post-decision and non-interventional.  It never
changes the selected diagnosis, candidate scores, retrieval, grounding, or
safety decision.  It converts existing pipeline state into auditable,
claim-level evidence mappings.

Reference design:
    SourceCheckup, Nature Communications (2025)
    https://www.nature.com/articles/s41467-025-58551-6

Important safety properties
---------------------------
* Similarity or reranker scores are used only to rank evidence candidates.
* A high score is never treated by itself as proof of a clinical claim.
* Patient facts are verified against the supplied case text.
* Diagnostic claims inherit the decision of the existing Evidence Judge.
* KG paths are shown only when they were returned by the graph layer.
* Unsupported and contradictory claims remain visible.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import math
import re
from app.rag.common_utils import _low, _safe_float

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

EXPLAINABILITY_VERSION = "v7.15-clinical-safety-candidate-trace-sourcecheckup"
SOURCECHECKUP_REFERENCE = "https://www.nature.com/articles/s41467-025-58551-6"

_MEDICAL_SAFETY_NOTES = [
    "This is decision-support only and not a final clinical diagnosis.",
    "Neuroimaging and physician assessment are required for acute stroke decisions.",
    "Claim support describes support in the available evidence, not clinical certainty.",
]

_CANDIDATE_LABELS = {
    "acute_ischemic_stroke": "Acute ischemic stroke",
    "tia": "Transient ischemic attack / TIA",
    "intracerebral_hemorrhage": "Intracerebral hemorrhage / hemorrhagic stroke",
    "subarachnoid_hemorrhage": "Subarachnoid hemorrhage / acute intracranial hemorrhage",
    "hemorrhagic_stroke": "Hemorrhagic stroke / intracranial hemorrhage",
    "posterior_circulation_stroke": "Posterior circulation stroke / vertebrobasilar ischemic stroke",
    "brainstem_infarction": "Brainstem infarction / posterior circulation stroke",
    "cerebellar_infarction": "Cerebellar infarction / posterior circulation stroke",
    "basilar_artery_occlusion": "Basilar artery occlusion / posterior circulation stroke",
    "large_vessel_occlusion": "Large vessel occlusion / acute ischemic stroke",
    "lacunar_infarction": "Lacunar infarction / small vessel ischemic stroke",
    "anterior_circulation_stroke": "Anterior circulation stroke / MCA-carotid territory ischemic stroke",
}

# Used only for evidence-family alignment/contradiction checks.  These are not
# diagnostic rules and never modify the selected candidate.
_FAMILY_TERMS = {
    "ischemic": [
        "ischemic", "ischaemic", "infarct", "infarction", "thrombolysis",
        "thrombectomy", "large vessel occlusion", "mca occlusion",
        "posterior circulation", "vertebrobasilar", "cerebellar infarction",
    ],
    "hemorrhagic": [
        "hemorrhage", "haemorrhage", "hemorrhagic", "haemorrhagic",
        "hematoma", "haematoma", "subarachnoid", "intracerebral", "sah", "ich",
    ],
    "tia": [
        "transient ischemic attack", "transient ischaemic attack", "tia",
        "transient neurological deficit", "resolved neurological deficit",
    ],
}

_SIGNAL_MESSAGES = {
    "is_acute": "The case contains an acute or sudden temporal presentation.",
    "has_motor": "The case contains a focal motor neurological finding.",
    "has_language": "The case contains a language or speech neurological finding.",
    "is_posterior": "The case contains posterior-circulation-compatible findings.",
    "is_hemorrhage": "The case contains hemorrhage-relevant findings.",
    "asks_complication": "The case concerns a post-stroke complication.",
    "is_chronic_or_general": "The case contains chronic or general non-acute context.",
    "is_mimic_or_nonstroke": "The case contains a possible mimic or non-stroke context.",
}

_NEGATIONS = [
    "no", "not", "without", "denies", "denied", "negative for", "absence of",
    "لا", "بدون", "ينفي", "لا يوجد", "غياب",
]


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    NOT_VERIFIABLE = "not_verifiable"


class ClaimType(str, Enum):
    PATIENT_FACT = "patient_fact"
    MEDICAL_KNOWLEDGE = "medical_knowledge"
    DIAGNOSTIC = "diagnostic"
    EXCLUSION = "exclusion"
    TREATMENT_RECOMMENDATION = "treatment_or_recommendation"
    UNCERTAINTY = "uncertainty"
    SYSTEM_PROVENANCE = "system_provenance"


@dataclass
class ClaimRecord:
    claim_id: str
    claim_text: str
    claim_type: str
    support_status: str = ClaimStatus.NOT_VERIFIABLE.value
    support_score: Optional[float] = None
    patient_evidence: List[Dict[str, Any]] = field(default_factory=list)
    document_evidence: List[Dict[str, Any]] = field(default_factory=list)
    kg_evidence: List[Dict[str, Any]] = field(default_factory=list)
    contradicting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    verification_reason: str = ""
    verification_method: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_explainability_cfg() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "enabled": True,
        "methodology": "sourcecheckup",
        "max_evidence_per_claim": 5,
        "max_patient_fact_claims": 8,
        "support_threshold": 0.66,
        "partial_support_threshold": 0.45,
        "contradiction_threshold": 0.55,
        "partial_support_weight": 0.5,
        "use_llm_verifier": False,
    }
    if yaml is None:
        return defaults
    try:
        cfg_path = Path("config.yaml")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        user_cfg = cfg.get("explainability", {}) or {}
        defaults.update(user_cfg)
    except Exception:
        pass
    return defaults


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _as_mapping(x: Any) -> Dict[str, Any]:
    return dict(x) if isinstance(x, Mapping) else {}


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _dedup_dicts(items: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        d = dict(item)
        marker = str(d.get("evidence_id") or d.get("path_id") or d.get("claim_id") or json.dumps(d, ensure_ascii=False, sort_keys=True, default=str))
        if marker in seen:
            continue
        seen.add(marker)
        out.append(d)
    return out


def _candidate_key(item: Any) -> Optional[str]:
    if isinstance(item, Mapping):
        for key in ("candidate_key", "key", "selected_candidate"):
            if item.get(key):
                return str(item.get(key))
        value = item.get("candidate")
        if value:
            low = _low(value)
            for k, label in _CANDIDATE_LABELS.items():
                if low == _low(label):
                    return k
            return str(value)
    if isinstance(item, str):
        low = _low(item)
        for k, label in _CANDIDATE_LABELS.items():
            if low in {_low(k), _low(label)}:
                return k
        return item
    return None


def _candidate_label(item: Any) -> str:
    key = _candidate_key(item)
    if key in _CANDIDATE_LABELS:
        return _CANDIDATE_LABELS[str(key)]
    if isinstance(item, Mapping):
        return str(item.get("candidate") or item.get("label") or key or "Unknown candidate")
    return str(item or "Unknown candidate").replace("_", " ").strip().title()


def _source_text(source: Mapping[str, Any]) -> str:
    values = [
        source.get("title"), source.get("text"), source.get("snippet"),
        source.get("graph_disease"), source.get("disease"),
        source.get("graph_category"), source.get("category"),
        source.get("candidate_label"), source.get("candidate_key"),
    ]
    return " ".join(str(v) for v in values if v).strip()


def _source_rank(source: Mapping[str, Any]) -> float:
    """Rank evidence candidates; this is not a truth/entailment score."""
    medcpt = _safe_float(source.get("medcpt_score"), 0.0)
    if medcpt > 1.5:
        medcpt = min(medcpt / 8.0, 1.0)
    hybrid = _safe_float(source.get("hybrid_score", source.get("score")), 0.0)
    graph = _safe_float(source.get("graph_score"), 0.0)
    grounding = _safe_float(source.get("_grounding_strength", source.get("grounding_strength")), 0.0)
    return round(0.35 * min(max(medcpt, 0.0), 1.0) + 0.30 * min(max(hybrid, 0.0), 1.0) + 0.20 * min(max(graph, 0.0), 1.0) + 0.15 * min(max(grounding, 0.0), 1.0), 4)


def _normalize_evidence(source: Mapping[str, Any], *, role: str = "candidate") -> Dict[str, Any]:
    text = str(source.get("text") or source.get("snippet") or "").strip()
    evidence_id = str(source.get("evidence_id") or source.get("chunk_id") or source.get("pmid") or source.get("doi") or source.get("title") or "unknown")
    is_kg = bool(source.get("from_kg") or source.get("graph_score") is not None or "kg" in _low(source.get("retrieval_channel")))
    return {
        "evidence_id": evidence_id,
        "document_id": str(source.get("document_id") or source.get("pmid") or source.get("doi") or source.get("source") or "unknown"),
        "chunk_id": source.get("chunk_id"),
        "pmid": source.get("pmid"),
        "doi": source.get("doi"),
        "title": source.get("title"),
        "source_name": source.get("source"),
        "evidence_text": text[:1000],
        "evidence_role": role,
        "retrieval_channel": source.get("retrieval_channel"),
        "retrieval_views": list(source.get("retrieval_views") or []),
        "dense_score": source.get("dense_score"),
        "bm25_score": source.get("bm25_score"),
        "hybrid_score": source.get("hybrid_score", source.get("score")),
        "medcpt_score": source.get("medcpt_score"),
        "graph_score": source.get("graph_score"),
        "grounding_strength": source.get("_grounding_strength", source.get("grounding_strength")),
        "candidate_rank_score": _source_rank(source),
        "grounding_result": source.get("grounding_result"),
        "from_kg": is_kg,
        "graph_disease": source.get("graph_disease") or source.get("disease"),
        "graph_category": source.get("graph_category") or source.get("category"),
        "kg_paths": list(source.get("kg_paths") or []),
        "evidence_match": dict(source.get("_evidence_match") or source.get("evidence_match") or {}),
    }


def _contains_phrase(text: str, phrase: str) -> bool:
    t = _low(text)
    p = _low(phrase)
    if not p:
        return False
    if re.search(r"[\u0600-\u06FF]", p):
        return p in t
    if len(p) <= 4 and re.fullmatch(r"[a-z0-9]+", p):
        return re.search(rf"\b{re.escape(p)}\b", t) is not None
    return p in t


def _phrase_is_negated(case_text: str, phrase: str, window_chars: int = 90) -> bool:
    low = _low(case_text)
    phrase_low = _low(phrase)
    if not phrase_low:
        return False
    for match in re.finditer(re.escape(phrase_low), low):
        left = low[max(0, match.start() - window_chars):match.start()]
        if any(re.search(rf"(?:^|\s){re.escape(neg)}(?:\s|$)", left) for neg in _NEGATIONS):
            return True
    return False


def _family_for_candidate(candidate_key: Optional[str], label: str) -> str:
    key = _low(candidate_key)
    text = _low(label)
    if key == "tia" or "transient ischemic" in text:
        return "tia"
    if key in {"intracerebral_hemorrhage", "subarachnoid_hemorrhage", "hemorrhagic_stroke"} or any(t in text for t in _FAMILY_TERMS["hemorrhagic"]):
        return "hemorrhagic"
    return "ischemic"


def _source_family(source: Mapping[str, Any]) -> Optional[str]:
    text = _low(_source_text(source))
    hits = {family: sum(1 for term in terms if _contains_phrase(text, term)) for family, terms in _FAMILY_TERMS.items()}
    if not hits or max(hits.values()) <= 0:
        return None
    return max(hits, key=hits.get)


def _compact_upstream_signals(signals: Mapping[str, Any], intent: Mapping[str, Any]) -> Dict[str, Any]:
    active = [name for name in _SIGNAL_MESSAGES if bool(signals.get(name))]
    return {
        "active_signals": active,
        "scores": dict(signals.get("scores") or {}),
        "time_context": dict(signals.get("time_context") or {}),
        "rule_hits": dict(signals.get("rule_hits") or {}),
        "model": dict(signals.get("model") or {}),
        "clinical_intent": dict(intent or {}),
        "source": signals.get("source"),
    }


def _candidate_judgment_rows(candidates: Sequence[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in candidates or []:
        if isinstance(item, Mapping):
            rows.append(dict(item))
        else:
            rows.append({"candidate": str(item), "candidate_key": _candidate_key(item), "decision": "unknown", "support_score": 0.0})
    return rows


def _find_selected_verification(selected: Mapping[str, Any], verifications: Mapping[str, Any]) -> Dict[str, Any]:
    keys = [
        selected.get("candidate_key"), selected.get("candidate"),
        _candidate_label(selected), _candidate_key(selected),
    ]
    for key in keys:
        if key is not None and isinstance(verifications.get(str(key)), Mapping):
            return dict(verifications[str(key)])
    return {}


def _extract_rule_hits(signals: Mapping[str, Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    rule_hits = signals.get("rule_hits") or {}
    if not isinstance(rule_hits, Mapping):
        return out
    for category, values in rule_hits.items():
        for value in _as_list(values):
            cue = str(value).strip()
            if cue:
                out.append((str(category), cue))
    return out


def _patient_span(case_text: str, cue: str) -> Dict[str, Any]:
    low = _low(case_text)
    cue_low = _low(cue)
    idx = low.find(cue_low)
    if idx < 0:
        return {"source": "patient_case", "matched_cue": cue, "text_span": "", "start": None, "end": None}
    start = max(0, idx - 45)
    end = min(len(case_text), idx + len(cue) + 45)
    return {"source": "patient_case", "matched_cue": cue, "text_span": case_text[start:end], "start": idx, "end": idx + len(cue)}


def _rank_matching_sources(
    claim_text: str,
    sources: Sequence[Mapping[str, Any]],
    *,
    candidate_label: str = "",
    max_items: int = 5,
) -> List[Dict[str, Any]]:
    claim_tokens = {t for t in re.findall(r"[a-zA-Z0-9\u0600-\u06FF]+", _low(claim_text)) if len(t) > 2}
    candidate_terms = {t for t in re.findall(r"[a-zA-Z0-9]+", _low(candidate_label)) if len(t) > 2}
    ranked: List[Tuple[float, Mapping[str, Any]]] = []
    for source in sources or []:
        if not isinstance(source, Mapping):
            continue
        text = _low(_source_text(source))
        if not text:
            continue
        overlap = sum(1 for token in claim_tokens if token in text)
        candidate_overlap = sum(1 for token in candidate_terms if token in text)
        evidence_match = source.get("_evidence_match") or source.get("evidence_match") or {}
        structured_bonus = 1.0 if any(bool(v) for v in _as_mapping(evidence_match).values()) else 0.0
        score = _source_rank(source) + 0.05 * overlap + 0.08 * candidate_overlap + 0.15 * structured_bonus
        if overlap > 0 or candidate_overlap > 0 or structured_bonus > 0:
            ranked.append((score, source))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [_normalize_evidence(source) for _, source in ranked[:max_items]]


# [تم حذف كود ميت مكرر غير مستخدم: _build_patient_fact_claims (old) — السطر الأصلي 416-461]
def _build_diagnostic_claim(
    final_answer: str,
    selected: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> ClaimRecord:
    label = _candidate_label(selected) if selected else final_answer
    claim = ClaimRecord(
        claim_id="claim_diagnosis_1",
        claim_text=f"The final system diagnosis is {final_answer}.",
        claim_type=ClaimType.DIAGNOSTIC.value,
        verification_method="existing_evidence_judge_decision",
    )
    if _low(final_answer) == "evidence is insufficient" or not selected:
        claim.support_status = ClaimStatus.NOT_VERIFIABLE.value
        claim.support_score = 0.0
        claim.verification_reason = "No candidate passed the Evidence Judge and safety requirements."
        claim.missing_information = ["Sufficient grounded evidence for a supported diagnostic candidate."]
        return claim

    score = _safe_float(selected.get("support_score"), 0.0)
    decision = _low(selected.get("decision"))
    penalty = _safe_float(selected.get("contradiction_penalty"), 0.0)
    support_threshold = _safe_float(cfg.get("support_threshold"), 0.66)
    partial_threshold = _safe_float(cfg.get("partial_support_threshold"), 0.45)
    contradiction_threshold = _safe_float(cfg.get("contradiction_threshold"), 0.55)

    supporting_raw = [x for x in _as_list(selected.get("supporting_sources")) if isinstance(x, Mapping)]
    conflicting_raw = [x for x in _as_list(selected.get("conflicting_sources")) if isinstance(x, Mapping)]
    if not supporting_raw:
        supporting_raw = [x for x in sources if isinstance(x, Mapping)]

    max_items = _safe_int(cfg.get("max_evidence_per_claim"), 5)
    claim.document_evidence = _rank_matching_sources(claim.claim_text, supporting_raw, candidate_label=label, max_items=max_items)
    claim.kg_evidence = [e for e in claim.document_evidence if e.get("from_kg")]
    claim.document_evidence = [e for e in claim.document_evidence if not e.get("from_kg")]
    claim.contradicting_evidence = [_normalize_evidence(x, role="contradicting") for x in conflicting_raw[:max_items]]
    claim.support_score = round(score, 4)

    if penalty >= contradiction_threshold and conflicting_raw and decision == "rejected":
        claim.support_status = ClaimStatus.CONTRADICTED.value
        claim.verification_reason = "The Evidence Judge rejected the candidate because conflicting evidence exceeded the contradiction threshold."
    elif decision == "supported" and score >= support_threshold and (supporting_raw or claim.kg_evidence):
        claim.support_status = ClaimStatus.SUPPORTED.value
        claim.verification_reason = "The existing Evidence Judge marked the candidate supported using retrieved evidence, KG verification, and grounding."
    elif decision in {"supported", "weak"} or score >= partial_threshold:
        claim.support_status = ClaimStatus.PARTIALLY_SUPPORTED.value
        claim.verification_reason = "The candidate has some evidence support but does not meet all full-support conditions."
    elif conflicting_raw and penalty > 0:
        claim.support_status = ClaimStatus.CONTRADICTED.value
        claim.verification_reason = "Available evidence contains a stronger conflicting diagnostic family."
    else:
        claim.support_status = ClaimStatus.UNSUPPORTED.value
        claim.verification_reason = "No sufficient claim-level evidence passed the existing Evidence Judge thresholds."
    return claim


def _build_kg_claim(selected: Mapping[str, Any], verification: Mapping[str, Any], cfg: Mapping[str, Any]) -> Optional[ClaimRecord]:
    if not selected:
        return None
    label = _candidate_label(selected)
    docs = [x for x in _as_list(verification.get("docs")) if isinstance(x, Mapping)]
    paths = [x for x in _as_list(verification.get("kg_paths")) if isinstance(x, Mapping)]
    if not paths:
        paths = [p for d in docs for p in _as_list(d.get("kg_paths")) if isinstance(p, Mapping)]
    source_count = _safe_int(verification.get("source_count"), len(docs))
    graph_score = _safe_float(verification.get("kg_support_score"), 0.0)
    claim = ClaimRecord(
        claim_id="claim_kg_1",
        claim_text=f"The knowledge graph contains evidence linked to {label}.",
        claim_type=ClaimType.SYSTEM_PROVENANCE.value,
        support_score=round(graph_score, 4),
        kg_evidence=[_normalize_evidence(d) for d in docs[:_safe_int(cfg.get("max_evidence_per_claim"), 5)]],
        verification_method="candidate_specific_kg_verification",
    )
    if source_count > 0 and paths:
        claim.support_status = ClaimStatus.SUPPORTED.value
        claim.verification_reason = "Candidate-specific Neo4j retrieval returned sources and auditable KG paths."
    elif source_count > 0:
        claim.support_status = ClaimStatus.PARTIALLY_SUPPORTED.value
        claim.verification_reason = "KG-linked documents were returned, but no structured path metadata was available."
    else:
        claim.support_status = ClaimStatus.UNSUPPORTED.value
        claim.verification_reason = "Candidate-specific KG verification returned no source."
    return claim


def _collect_kg_paths(sources: Sequence[Mapping[str, Any]], verification: Mapping[str, Any]) -> List[Dict[str, Any]]:
    paths: List[Mapping[str, Any]] = []
    paths.extend(x for x in _as_list(verification.get("kg_paths")) if isinstance(x, Mapping))
    for source in sources or []:
        if isinstance(source, Mapping):
            paths.extend(x for x in _as_list(source.get("kg_paths")) if isinstance(x, Mapping))
    return _dedup_dicts(paths)[:30]


def _missing_clinical_information(case_text: str, signals: Mapping[str, Any], final_answer: str) -> List[Dict[str, str]]:
    low = _low(case_text)
    items: List[Tuple[str, Sequence[str], str]] = [
        ("last_known_well_or_exact_onset", ["last known well", "onset", "started", "began", "minutes", "hours", "منذ", "خلال"], "Exact onset information is not documented."),
        ("neuroimaging", ["ct", "cta", "mri", "dwi", "تصوير", "طبقي", "رنين"], "No neuroimaging result is documented."),
        ("blood_glucose", ["glucose", "blood sugar", "hypoglycemia", "سكر"], "Blood glucose is not documented, so a metabolic mimic may remain unresolved."),
        ("nihss", ["nihss"], "NIHSS is not documented."),
        ("anticoagulant_status", ["warfarin", "apixaban", "rivaroxaban", "dabigatran", "anticoagulant", "anticoagulation", "مميع"], "Anticoagulant use is not documented."),
    ]
    missing: List[Dict[str, str]] = []
    acute_relevant = bool(signals.get("is_acute") or signals.get("has_motor") or signals.get("has_language") or signals.get("is_posterior") or signals.get("is_hemorrhage"))
    if not acute_relevant and _low(final_answer) == "evidence is insufficient":
        return missing
    for key, cues, reason in items:
        if not any(cue in low for cue in cues):
            missing.append({"field": key, "status": "not_documented", "reason": reason})
    return missing



def _score_breakdown(row: Mapping[str, Any]) -> Dict[str, float]:
    """Expose only score components already produced by Evidence Judge."""
    return {
        "textual_support": _safe_float(row.get("textual_support"), 0.0),
        "kg_support": _safe_float(row.get("kg_support"), 0.0),
        "grounding_support": _safe_float(row.get("grounding_support"), 0.0),
        "rerank_strength": _safe_float(row.get("rerank_strength"), 0.0),
        "source_diversity": _safe_float(row.get("source_diversity"), 0.0),
        "intent_compatibility": _safe_float(row.get("intent_compatibility"), 0.0),
        "contradiction_penalty": _safe_float(row.get("contradiction_penalty"), 0.0),
    }


def _evidence_ids(values: Any) -> List[str]:
    ids: List[str] = []
    for item in _as_list(values):
        if not isinstance(item, Mapping):
            continue
        evidence_id = item.get("evidence_id") or item.get("chunk_id") or item.get("pmid") or item.get("doi") or item.get("title")
        if evidence_id is not None:
            ids.append(str(evidence_id))
    return list(dict.fromkeys(ids))


def _candidate_comparison(rows: Sequence[Mapping[str, Any]], selected_key: Optional[str]) -> List[Dict[str, Any]]:
    """Build the auditable candidate-level decision table.

    Ranking is based on the final support score emitted by Evidence Judge.  This
    function never recalculates the score and never changes the selected result.
    """
    output: List[Dict[str, Any]] = []
    for row in rows:
        key = _candidate_key(row)
        score = _safe_float(row.get("support_score"), 0.0)
        selected = bool(key and selected_key and _low(key) == _low(selected_key))
        decision = str(row.get("decision") or "unknown")
        reason = str(row.get("reason") or "").strip()
        breakdown = _score_breakdown(row)
        output.append({
            "candidate": _candidate_label(row),
            "candidate_key": key,
            "candidate_label": _candidate_label(row),
            "selected": selected,
            "decision": decision,
            "final_support_score": score,
            "support_score": score,  # backward compatibility
            "support_count": _safe_int(row.get("support_count"), 0),
            "grounding_ok": bool(row.get("grounding_ok")),
            "score_breakdown": breakdown,
            # flattened legacy columns used by the current UI
            **breakdown,
            "reason": reason,
            "selection_reasons": [],
            "weakening_reasons": [],
            "rejection_reasons": [],
            "why_not_selected": [],
            "supporting_source_ids": _evidence_ids(row.get("supporting_sources")),
            "conflicting_source_ids": _evidence_ids(row.get("conflicting_sources")),
        })

    output.sort(key=lambda x: x["final_support_score"], reverse=True)
    selected_score = next((x["final_support_score"] for x in output if x["selected"]), 0.0)

    for rank, item in enumerate(output, start=1):
        item["rank"] = rank
        breakdown = item["score_breakdown"]
        if item["selected"]:
            item["selection_reasons"].append(
                f"Evidence Judge selected this candidate with final support score {item['final_support_score']:.4f}."
            )
            if rank == 1:
                item["selection_reasons"].append("It ranked first among the evaluated candidates by final support score.")
            if item["support_count"]:
                item["selection_reasons"].append(
                    f"It had {item['support_count']} supporting evidence source(s) after candidate-level filtering."
                )
            if item["grounding_ok"]:
                item["selection_reasons"].append("It passed the Evidence Judge grounding requirement.")
            if breakdown["textual_support"] > 0:
                item["selection_reasons"].append(
                    f"Textual evidence support contributed {breakdown['textual_support']:.4f}."
                )
            if breakdown["kg_support"] > 0:
                item["selection_reasons"].append(
                    f"Knowledge-graph support contributed {breakdown['kg_support']:.4f}."
                )
            if item["reason"]:
                item["selection_reasons"].append(f"Evidence Judge reason: {item['reason']}")
        else:
            difference = max(selected_score - item["final_support_score"], 0.0)
            item["score_difference_from_selected"] = round(difference, 4)
            if difference > 0:
                item["why_not_selected"].append(
                    f"Its final support score was {difference:.4f} lower than the selected candidate."
                )
            if item["decision"] != "supported":
                item["why_not_selected"].append(f"Evidence Judge classified it as {item['decision']}.")
            if breakdown["contradiction_penalty"] > 0:
                item["weakening_reasons"].append(
                    f"Contradiction penalty was {breakdown['contradiction_penalty']:.4f}."
                )
            if not item["grounding_ok"]:
                item["weakening_reasons"].append("It did not pass the candidate-level grounding requirement.")
            if item["reason"]:
                item["rejection_reasons"].append(f"Evidence Judge reason: {item['reason']}")
            item["why_not_selected"].extend(item["weakening_reasons"])
            item["why_not_selected"].extend(item["rejection_reasons"])
            item["why_not_selected"] = list(dict.fromkeys(item["why_not_selected"]))
    return output


def _metrics(claims: Sequence[ClaimRecord], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    total = len(claims)
    counts = {status.value: sum(1 for c in claims if c.support_status == status.value) for status in ClaimStatus}
    verifiable = total - counts[ClaimStatus.NOT_VERIFIABLE.value]
    supported = counts[ClaimStatus.SUPPORTED.value]
    partial = counts[ClaimStatus.PARTIALLY_SUPPORTED.value]
    contradicted = counts[ClaimStatus.CONTRADICTED.value]
    cited_claims = sum(1 for c in claims if c.patient_evidence or c.document_evidence or c.kg_evidence or c.contradicting_evidence)
    partial_weight = _safe_float(cfg.get("partial_support_weight"), 0.5)
    claim_support_rate = supported / verifiable if verifiable else 0.0
    weighted_support_rate = (supported + partial_weight * partial) / verifiable if verifiable else 0.0
    citation_coverage = cited_claims / verifiable if verifiable else 0.0
    contradiction_rate = contradicted / verifiable if verifiable else 0.0
    return {
        "total_claims": total,
        "verifiable_claims": verifiable,
        "supported_claims": supported,
        "partially_supported_claims": partial,
        "unsupported_claims": counts[ClaimStatus.UNSUPPORTED.value],
        "contradicted_claims": contradicted,
        "not_verifiable_claims": counts[ClaimStatus.NOT_VERIFIABLE.value],
        "claim_support_rate": round(claim_support_rate, 4),
        "weighted_claim_support_rate": round(weighted_support_rate, 4),
        "citation_coverage": round(citation_coverage, 4),
        "contradiction_rate": round(contradiction_rate, 4),
        "fully_supported_response": bool(verifiable > 0 and supported == verifiable),
        "partial_support_weight": partial_weight,
    }


def _confidence_level(final_answer: str, selected: Mapping[str, Any], grounding: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    if _low(final_answer) == "evidence is insufficient" or not selected:
        return "low"
    score = _safe_float(selected.get("support_score"), 0.0)
    top = _safe_float(grounding.get("top_strength"), 0.0)
    coverage = _safe_float(metrics.get("weighted_claim_support_rate"), 0.0)
    if score >= 0.80 and top >= 0.65 and coverage >= 0.80:
        return "high"
    if score >= 0.60 and top >= 0.45 and coverage >= 0.50:
        return "moderate"
    return "low"


def _grounding_explanation(grounding: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "allow_llm": bool(grounding.get("allow_llm")),
        "reason": grounding.get("reason"),
        "top_strength": _safe_float(grounding.get("top_strength"), 0.0),
        "support_count": _safe_int(grounding.get("support_count"), 0),
        "avg_strength_top3": _safe_float(grounding.get("avg_strength_top3"), 0.0),
        "interpretation": "Grounding is a pipeline-level evidence sufficiency gate; it is not claim entailment.",
    }


def _conflict_analysis(rows: Sequence[Mapping[str, Any]], selected_key: Optional[str]) -> Dict[str, Any]:
    selected = next((r for r in rows if selected_key and _low(_candidate_key(r)) == _low(selected_key)), {})
    conflicts = [
        _normalize_evidence(x, role="contradicting")
        for x in _as_list(selected.get("conflicting_sources"))
        if isinstance(x, Mapping)
    ]
    return {
        "selected_candidate": selected_key,
        "contradiction_penalty": _safe_float(selected.get("contradiction_penalty"), 0.0),
        "conflicting_sources": conflicts,
        "has_explicit_conflict_sources": bool(conflicts),
    }



def _claim_evidence_index(claims: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for claim in claims:
        claim_id = str(claim.get("claim_id") or "")
        status = str(claim.get("support_status") or ClaimStatus.NOT_VERIFIABLE.value)
        for field_name in ("document_evidence", "kg_evidence", "contradicting_evidence"):
            for evidence in _as_list(claim.get(field_name)):
                if not isinstance(evidence, Mapping):
                    continue
                evidence_id = str(evidence.get("evidence_id") or evidence.get("chunk_id") or evidence.get("pmid") or evidence.get("doi") or evidence.get("title") or "unknown")
                bucket = index.setdefault(evidence_id, {"claim_ids": [], "statuses": [], "claim_texts": []})
                if claim_id and claim_id not in bucket["claim_ids"]:
                    bucket["claim_ids"].append(claim_id)
                    bucket["statuses"].append(status)
                    bucket["claim_texts"].append(str(claim.get("claim_text") or ""))
    return index


def _supporting_evidence(
    selected: Mapping[str, Any],
    claim_dicts: Sequence[Mapping[str, Any]],
    selected_label: str,
    cfg: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    claim_index = _claim_evidence_index(claim_dicts)
    docs = [x for x in _as_list(selected.get("supporting_sources")) if isinstance(x, Mapping)]
    docs.extend(x for x in _as_list(selected.get("candidate_kg_sources")) if isinstance(x, Mapping))
    normalized = _dedup_dicts(_normalize_evidence(x, role="supporting") for x in docs)
    normalized.sort(key=lambda x: _safe_float(x.get("candidate_rank_score"), 0.0), reverse=True)
    out: List[Dict[str, Any]] = []
    for rank, evidence in enumerate(normalized[:_safe_int(cfg.get("max_evidence_per_claim"), 5)], start=1):
        evidence_id = str(evidence.get("evidence_id") or "unknown")
        linked = claim_index.get(evidence_id, {"claim_ids": [], "statuses": [], "claim_texts": []})
        statuses = list(linked.get("statuses") or [])
        if ClaimStatus.CONTRADICTED.value in statuses:
            support_status = ClaimStatus.CONTRADICTED.value
        elif ClaimStatus.SUPPORTED.value in statuses:
            support_status = ClaimStatus.SUPPORTED.value
        elif ClaimStatus.PARTIALLY_SUPPORTED.value in statuses:
            support_status = ClaimStatus.PARTIALLY_SUPPORTED.value
        elif statuses:
            support_status = statuses[0]
        else:
            support_status = ClaimStatus.NOT_VERIFIABLE.value
        snippet = str(evidence.get("evidence_text") or "").strip()
        claim_texts = [x for x in linked.get("claim_texts", []) if x]
        why_relevant = (
            f"This evidence was retained by Evidence Judge for {selected_label} and is linked to: {claim_texts[0]}"
            if claim_texts
            else f"This evidence was retained as supporting evidence for {selected_label}; no claim-level link was verified."
        )
        out.append({
            "evidence_id": evidence_id,
            "rank": rank,
            "pmid": evidence.get("pmid"),
            "doi": evidence.get("doi"),
            "title": evidence.get("title"),
            "source_url": evidence.get("source_name"),
            "retrieval_channel": evidence.get("retrieval_channel"),
            "snippet": snippet,
            "exact_supporting_span": snippet,
            "supported_claim_ids": list(linked.get("claim_ids") or []),
            "why_relevant": why_relevant,
            "scores": {
                "dense_score": evidence.get("dense_score"),
                "bm25_score": evidence.get("bm25_score"),
                "hybrid_score": evidence.get("hybrid_score"),
                "medcpt_score": evidence.get("medcpt_score"),
                "graph_score": evidence.get("graph_score"),
                "grounding_strength": evidence.get("grounding_strength"),
            },
            # flattened aliases make Streamlit/CSV consumption easier
            "dense_score": evidence.get("dense_score"),
            "bm25_score": evidence.get("bm25_score"),
            "hybrid_score": evidence.get("hybrid_score"),
            "medcpt_score": evidence.get("medcpt_score"),
            "graph_score": evidence.get("graph_score"),
            "grounding_strength": evidence.get("grounding_strength"),
            "from_kg": bool(evidence.get("from_kg")),
            "support_status": support_status,
            "kg_paths": list(evidence.get("kg_paths") or []),
        })
    return out


def _decision_narrative(comparison: Sequence[Mapping[str, Any]], selected_label: str) -> str:
    selected = next((x for x in comparison if x.get("selected")), {})
    if not selected:
        return "لم يتمكن النظام من اختيار تشخيص مدعوم من المرشحين المتاحين."
    count = len(comparison)
    score = _safe_float(selected.get("final_support_score"), 0.0)
    parts = [
        f"درس النظام {count} تشخيصًا مرشحًا.",
        f"تم اختيار {selected_label} بدرجة دعم نهائية {score:.4f} بعد تقييم مكونات Evidence Judge الفعلية وخصم التعارضات.",
    ]
    alternatives = [x for x in comparison if not x.get("selected")]
    if alternatives:
        compared = "، ".join(
            f"{x.get('candidate_label')} ({_safe_float(x.get('final_support_score'), 0.0):.4f})"
            for x in alternatives[:4]
        )
        parts.append(f"درجات أبرز البدائل كانت: {compared}.")
    breakdown = _as_mapping(selected.get("score_breakdown"))
    active = [
        name for name, key in (
            ("الدعم النصي", "textual_support"),
            ("دعم Knowledge Graph", "kg_support"),
            ("Grounding", "grounding_support"),
            ("MedCPT reranking", "rerank_strength"),
            ("تنوع المصادر", "source_diversity"),
            ("توافق الحالة", "intent_compatibility"),
        ) if _safe_float(breakdown.get(key), 0.0) > 0
    ]
    if active:
        parts.append("العوامل التي ظهرت في حساب المرشح المختار: " + "، ".join(active) + ".")
    penalty = _safe_float(breakdown.get("contradiction_penalty"), 0.0)
    if penalty > 0:
        parts.append(f"تم احتساب عقوبة تعارض مقدارها {penalty:.4f} قبل اعتماد النتيجة النهائية.")
    return "\n\n".join(parts)


def _why_selected(
    comparison: Sequence[Mapping[str, Any]],
    selected_label: str,
    claims: Sequence[Mapping[str, Any]],
    supporting_evidence: Sequence[Mapping[str, Any]],
    grounding: Mapping[str, Any],
    kg_paths: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    selected = next((x for x in comparison if x.get("selected")), {})
    patient_claims = [
        str(c.get("claim_text")) for c in claims
        if c.get("claim_type") == ClaimType.PATIENT_FACT.value
        and c.get("support_status") in {ClaimStatus.SUPPORTED.value, ClaimStatus.PARTIALLY_SUPPORTED.value}
    ]
    strongest_sources = [
        {
            "evidence_id": e.get("evidence_id"),
            "pmid": e.get("pmid"),
            "title": e.get("title"),
            "supported_claim_ids": e.get("supported_claim_ids", []),
            "medcpt_score": e.get("medcpt_score"),
            "hybrid_score": e.get("hybrid_score"),
            "grounding_strength": e.get("grounding_strength"),
        }
        for e in list(supporting_evidence)[:3]
    ]
    advantages = []
    selected_score = _safe_float(selected.get("final_support_score"), 0.0)
    for alt in comparison:
        if alt.get("selected"):
            continue
        diff = selected_score - _safe_float(alt.get("final_support_score"), 0.0)
        advantages.append({
            "over_candidate": alt.get("candidate_label"),
            "score_advantage": round(diff, 4),
        })
    selected_breakdown = _as_mapping(selected.get("score_breakdown"))
    return {
        "summary": _decision_narrative(comparison, selected_label),
        "main_reasons": list(selected.get("selection_reasons") or []),
        "strongest_clinical_signals": patient_claims[:6],
        "strongest_supporting_sources": strongest_sources,
        "score_advantages": advantages,
        "kg_support_summary": (
            f"KG support score was {_safe_float(selected_breakdown.get('kg_support'), 0.0):.4f}; {len(kg_paths)} auditable path(s) were preserved."
            if _safe_float(selected_breakdown.get("kg_support"), 0.0) > 0 or kg_paths
            else "No candidate-specific KG support was available."
        ),
        "grounding_summary": (
            f"Grounding reason={grounding.get('reason')}; top_strength={_safe_float(grounding.get('top_strength'), 0.0):.4f}; "
            f"support_count={_safe_int(grounding.get('support_count'), 0)}."
        ),
        "contradictions_considered": list(selected.get("conflicting_source_ids") or []),
    }


def _alternative_explanations(comparison: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    selected_score = next((_safe_float(x.get("final_support_score"), 0.0) for x in comparison if x.get("selected")), 0.0)
    out: List[Dict[str, Any]] = []
    for row in comparison:
        if row.get("selected"):
            continue
        out.append({
            "candidate": row.get("candidate_label"),
            "candidate_key": row.get("candidate_key"),
            "rank": row.get("rank"),
            "decision": row.get("decision"),
            "final_support_score": row.get("final_support_score"),
            "score_difference_from_selected": round(max(selected_score - _safe_float(row.get("final_support_score"), 0.0), 0.0), 4),
            "why_not_selected": list(row.get("why_not_selected") or []),
            "missing_support": [
                "No supporting source remained after Evidence Judge filtering."
            ] if _safe_int(row.get("support_count"), 0) == 0 else [],
            "conflicting_evidence": list(row.get("conflicting_source_ids") or []),
            "score_breakdown": dict(row.get("score_breakdown") or {}),
        })
    return out


def _uncertainty_summary(
    confidence: str,
    uncertainty_reasons: Sequence[str],
    missing: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
    conflicts: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "level": confidence,
        "reasons": list(uncertainty_reasons),
        "missing_clinical_information": [dict(x) for x in missing],
        "unresolved_conflicts": list(conflicts.get("conflicting_sources") or []),
        "unsupported_claims": [dict(c) for c in claims if c.get("support_status") == ClaimStatus.UNSUPPORTED.value],
        "not_verifiable_claims": [dict(c) for c in claims if c.get("support_status") == ClaimStatus.NOT_VERIFIABLE.value],
    }


# [تم حذف كود ميت مكرر غير مستخدم: build_explanation (old) — السطر الأصلي 976-1159]
def flatten_explanation_for_csv(explanation: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten stable fields for evaluation/error-analysis CSV files."""
    metrics = _as_mapping(explanation.get("metrics"))
    grounding = _as_mapping(explanation.get("grounding_explanation"))
    conflict = _as_mapping(explanation.get("conflict_analysis"))
    safety = _as_mapping(explanation.get("safety_summary"))
    return {
        "explainability_version": explanation.get("explainability_version"),
        "final_answer": explanation.get("final_answer"),
        "selected_candidate": explanation.get("selected_candidate"),
        "confidence_level": explanation.get("confidence_level"),
        "support_score": _safe_float(explanation.get("support_score"), 0.0),
        "decision_narrative": str(explanation.get("decision_narrative") or ""),
        "candidate_count": len(_as_list(explanation.get("candidate_comparison"))),
        "supporting_evidence_count": len(_as_list(explanation.get("supporting_evidence"))),
        "total_claims": _safe_int(metrics.get("total_claims"), 0),
        "supported_claims": _safe_int(metrics.get("supported_claims"), 0),
        "partially_supported_claims": _safe_int(metrics.get("partially_supported_claims"), 0),
        "unsupported_claims": _safe_int(metrics.get("unsupported_claims"), 0),
        "contradicted_claims": _safe_int(metrics.get("contradicted_claims"), 0),
        "not_verifiable_claims": _safe_int(metrics.get("not_verifiable_claims"), 0),
        "claim_support_rate": _safe_float(metrics.get("claim_support_rate"), 0.0),
        "weighted_claim_support_rate": _safe_float(metrics.get("weighted_claim_support_rate"), 0.0),
        "citation_coverage": _safe_float(metrics.get("citation_coverage"), 0.0),
        "contradiction_rate": _safe_float(metrics.get("contradiction_rate"), 0.0),
        "fully_supported_response": bool(metrics.get("fully_supported_response")),
        "grounding_allow_llm": bool(grounding.get("allow_llm")),
        "grounding_top_strength": _safe_float(grounding.get("top_strength"), 0.0),
        "grounding_support_count": _safe_int(grounding.get("support_count"), 0),
        "contradiction_penalty": _safe_float(conflict.get("contradiction_penalty"), 0.0),
        "conflicting_source_count": len(_as_list(conflict.get("conflicting_sources"))),
        "safety_blocked": bool(safety.get("blocked")),
        "claim_statuses_json": json.dumps(
            [
                {"claim_id": c.get("claim_id"), "claim_type": c.get("claim_type"), "support_status": c.get("support_status")}
                for c in _as_list(explanation.get("claims")) if isinstance(c, Mapping)
            ],
            ensure_ascii=False,
        ),
    }

# ============================================================
# v7.15 semantic patient claims and provisional subtype safety
# ============================================================
from app.rag.clinical_context import clinical_fact_spans as _v715_clinical_fact_spans


def _build_patient_fact_claims(case_text: str, signals: Mapping[str, Any], max_claims: int) -> List[ClaimRecord]:
    """Build atomic semantic facts, never standalone keyword claims."""
    facts = _v715_clinical_fact_spans(case_text)
    # Clinical completeness is more important than the legacy eight-keyword cap.
    limit = max(int(max_claims or 0), 12)
    claims: List[ClaimRecord] = []
    for fact in facts[:limit]:
        evidence = dict(fact.get("evidence") or {})
        evidence["polarity"] = fact.get("polarity") or "positive"
        claims.append(ClaimRecord(
            claim_id=f"claim_patient_{len(claims)+1}",
            claim_text=str(fact.get("claim_text") or "Patient fact."),
            claim_type=ClaimType.PATIENT_FACT.value,
            support_status=ClaimStatus.SUPPORTED.value,
            support_score=1.0,
            patient_evidence=[evidence],
            verification_reason=(
                "The negative finding is explicitly documented in the supplied case text."
                if fact.get("polarity") == "negative"
                else "The clinical fact is explicitly documented in the supplied case text."
            ),
            verification_method="semantic_case_fact_with_local_negation_and_temporal_scope",
        ))
    return claims


def _v715_metrics_from_claim_dicts(claims: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    clinical_claims = [
        c for c in claims
        if str(c.get("claim_type") or "") != ClaimType.SYSTEM_PROVENANCE.value
    ]
    total = len(clinical_claims)
    counts = {
        status.value: sum(1 for c in clinical_claims if c.get("support_status") == status.value)
        for status in ClaimStatus
    }
    verifiable = total - counts[ClaimStatus.NOT_VERIFIABLE.value]
    supported = counts[ClaimStatus.SUPPORTED.value]
    partial = counts[ClaimStatus.PARTIALLY_SUPPORTED.value]
    contradicted = counts[ClaimStatus.CONTRADICTED.value]
    cited = sum(1 for c in clinical_claims if any(c.get(k) for k in (
        "patient_evidence", "document_evidence", "kg_evidence", "contradicting_evidence"
    )))
    partial_weight = _safe_float(cfg.get("partial_support_weight"), 0.5)
    diagnostic_claims = [c for c in clinical_claims if c.get("claim_type") == ClaimType.DIAGNOSTIC.value]
    diagnostic_fully_supported = bool(
        diagnostic_claims
        and all(c.get("support_status") == ClaimStatus.SUPPORTED.value for c in diagnostic_claims)
    )
    return {
        "total_claims": total,
        "verifiable_claims": verifiable,
        "supported_claims": supported,
        "partially_supported_claims": partial,
        "unsupported_claims": counts[ClaimStatus.UNSUPPORTED.value],
        "contradicted_claims": contradicted,
        "not_verifiable_claims": counts[ClaimStatus.NOT_VERIFIABLE.value],
        "claim_support_rate": round(supported / verifiable, 4) if verifiable else 0.0,
        "weighted_claim_support_rate": round((supported + partial_weight * partial) / verifiable, 4) if verifiable else 0.0,
        "citation_coverage": round(cited / verifiable, 4) if verifiable else 0.0,
        "contradiction_rate": round(contradicted / verifiable, 4) if verifiable else 0.0,
        "fully_supported_response": bool(verifiable > 0 and supported == verifiable and diagnostic_fully_supported),
        "diagnostic_claim_fully_supported": diagnostic_fully_supported,
        "system_provenance_claims_excluded": sum(
            1 for c in claims if str(c.get("claim_type") or "") == ClaimType.SYSTEM_PROVENANCE.value
        ),
        "partial_support_weight": partial_weight,
    }


if "_v715_original_build_explanation" not in globals():
    _v715_original_build_explanation = build_explanation


def build_explanation(*args, **kwargs) -> Dict[str, Any]:
    output = _v715_original_build_explanation(*args, **kwargs)
    case_text = str(kwargs.get("case_text") or "")
    final_answer = str(kwargs.get("final_answer") or output.get("final_answer") or "")
    selected = _as_mapping(kwargs.get("selected_candidate"))
    cfg = _load_explainability_cfg()

    imaging_cues = ["ct", "cta", "ct perfusion", "mri", "dwi", "neuroimaging", "brain imaging", "طبقي", "رنين", "تصوير دماغ"]
    imaging_documented = any(cue in _low(case_text) for cue in imaging_cues)
    is_specific_subtype = bool(final_answer and _low(final_answer) != "evidence is insufficient")

    claims = [dict(c) for c in _as_list(output.get("claims")) if isinstance(c, Mapping)]
    if is_specific_subtype and not imaging_documented:
        for claim in claims:
            if claim.get("claim_type") != ClaimType.DIAGNOSTIC.value:
                continue
            # Literature/KG can support a provisional candidate, but it cannot
            # confirm the patient's ischemic-versus-hemorrhagic subtype without
            # patient neuroimaging.
            if claim.get("support_status") == ClaimStatus.SUPPORTED.value:
                claim["support_status"] = ClaimStatus.PARTIALLY_SUPPORTED.value
            claim["support_score"] = min(_safe_float(claim.get("support_score"), 0.0), 0.65)
            missing = list(claim.get("missing_information") or [])
            note = "Patient neuroimaging is required to confirm the stroke subtype and exclude hemorrhage."
            if note not in missing:
                missing.append(note)
            claim["missing_information"] = missing
            claim["verification_reason"] = (
                "The upstream Evidence Judge supports this as a provisional candidate, "
                "but patient-level stroke subtype confirmation requires neuroimaging."
            )
            claim["verification_method"] = "evidence_judge_plus_patient_imaging_requirement"

        output["claims"] = claims
        output["claim_evidence_map"] = claims
        output.setdefault("final_diagnosis", {})["provisional"] = True
        output["final_diagnosis"]["requires_neuroimaging_confirmation"] = True
        output["final_diagnosis"]["display_label"] = f"Provisional: {final_answer}"
        output.setdefault("uncertainty", {}).setdefault("reasons", [])
        reason = "The stroke subtype is provisional because no patient neuroimaging result is documented."
        if reason not in output["uncertainty"]["reasons"]:
            output["uncertainty"]["reasons"].append(reason)
        output.setdefault("medical_safety_notes", [])
        if reason not in output["medical_safety_notes"]:
            output["medical_safety_notes"].append(reason)

    metrics = _v715_metrics_from_claim_dicts(claims, cfg)
    output["metrics"] = metrics
    if is_specific_subtype and not imaging_documented:
        output["confidence_level"] = "moderate" if _safe_float(selected.get("support_score"), 0.0) >= 0.60 else "low"
        output.setdefault("final_diagnosis", {})["confidence_level"] = output["confidence_level"]
        if isinstance(output.get("confidence"), dict):
            output["confidence"]["level"] = output["confidence_level"]

    output["clinical_claim_metrics_note"] = (
        "Clinical claim metrics exclude system-provenance claims; support does not equal diagnostic certainty."
    )
    return output
