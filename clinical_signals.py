# -*- coding: utf-8 -*-
"""
Clinical signal detection layer for MedRAG.

Purpose:
- Keep the old keyword lists as explainable fallback/safety signals.
- Add an optional transformer-based multi-label clinical signal detector.
- Build dynamic query expansions that replace the one-size-fits-all DEFAULT_STROKE_EXPANSION behavior.

The transformer layer is OFF by default to keep the project runnable on machines
without downloaded models or GPU. Enable it with either:
    MEDRAG_USE_SIGNAL_MODEL=1
or config.yaml:
    clinical_signals:
      model_enabled: true
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional
from app.rag.common_utils import _contains_arabic, _load_cfg

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback if yaml is unavailable
    yaml = None

_AR_RE = re.compile(r"[\u0600-\u06FF]")
_WORD_RE = re.compile(r"[A-Za-z0-9]+|[\u0600-\u06FF]+", re.UNICODE)

# ---------------------------------------------------------------------------
# Rule-based vocabularies retained as fallback and explainability layer.
# These intentionally mirror the legacy retriever/pipeline clinical lists.
# ---------------------------------------------------------------------------
ACUTE_HINT_TERMS = [
    "acute", "sudden", "suddenly", "onset", "new onset", "acute onset",
    "abrupt", "abruptly", "within", "hour", "hours", "minute", "minutes",
    "min", "mins", "hr", "hrs", "today", "this morning", "this afternoon",
    "this evening", "last night", "same day", "nihss", "emergency",
    "prehospital", "fast", "act fast", "last known well", "wake-up stroke",
]

MOTOR_TERMS = [
    "weakness", "unilateral weakness", "arm weakness", "leg weakness",
    "hemiparesis", "hemiplegia", "facial droop", "facial weakness",
]

LANGUAGE_TERMS = [
    "aphasia", "expressive aphasia", "speech disturbance", "speech difficulty",
    "speech", "dysarthria", "slurred", "slurred speech",
]

POSTERIOR_TERMS = [
    "vertigo", "diplopia", "double vision", "ataxia", "dizziness",
    "gait difficulty", "difficulty walking", "unsteady gait", "truncal ataxia",
    "inability to walk", "nystagmus", "imbalance", "dysphagia", "hoarseness",
    "brainstem", "cerebellar", "cerebellum", "posterior circulation",
    "vertebrobasilar", "basilar", "lateral medullary", "wallenberg",
]

HEMORRHAGE_TERMS = [
    "hemorrhage", "haemorrhage", "bleeding", "intracerebral hemorrhage",
    "intracerebral haemorrhage", "subarachnoid hemorrhage",
    "subarachnoid haemorrhage", "sah", "ich", "thunderclap",
    "worst headache", "sudden severe headache", "vomiting",
    "loss of consciousness", "decreased consciousness", "reduced alertness",
    "worsening consciousness", "neck stiffness", "photophobia",
]

COMPLICATION_TOPICS = [
    "delirium", "depression", "poststroke depression", "post-stroke depression",
    "poststroke delirium", "post-stroke delirium", "seizure", "epilepsy",
    "poststroke seizure", "post-stroke seizure", "poststroke epilepsy",
    "post-stroke epilepsy", "spasticity", "dysphagia after stroke",
]

CHRONIC_OR_GENERAL_PENALTY_TERMS = [
    "cognitive impairment", "vascular cognitive impairment", "dementia", "alzheimer",
    "long-term", "long term", "survivor", "survivors", "rehabilitation",
    "risk factor", "risk factors", "genetic", "genome", "locus", "loci",
    "registry", "burden", "incidence", "prevalence", "epidemiology",
    "systematic review", "meta-analysis", "meta analysis", "chronic",
    "gradual", "gradually", "progressive", "progressively", "slowly",
    "for weeks", "for months", "for years", "over weeks", "over months",
    "over years", "several weeks", "several months", "several years",
]

MIMIC_TERMS = [
    "migraine", "aura", "seizure", "postictal", "hypoglycemia", "hypoglycaemia",
    "positional vertigo", "bppv", "vestibular neuritis", "syncope",
    "functional", "conversion", "panic",
]

AR_ACUTE_HINTS = [
    "مفاجئ", "فجأة", "حاد", "حادة", "خلال", "ساعة", "ساعات", "دقيقة", "دقائق",
    "ضعف", "شلل", "نصفي", "حبسة", "كلام", "نطق", "تلعثم", "تنميل",
    "دوخة", "دوار", "إغماء", "وعي", "وجه", "ميلان",
]

AR_POSTERIOR_HINTS = [
    "دوخة", "دوار", "ازدواج", "الرؤية", "ترنح", "توازن", "مشي", "مخيخ", "جذع الدماغ",
]

AR_HEMORRHAGE_HINTS = [
    "نزف", "نزيف", "صداع انفجاري", "أسوأ صداع", "قيء", "استفراغ", "فقدان وعي", "تيبس رقبة",
]

# Candidate labels for optional zero-shot classifier.
MODEL_LABELS = [
    "acute ischemic stroke presentation",
    "posterior circulation stroke presentation",
    "hemorrhagic stroke or intracranial hemorrhage presentation",
    "motor neurological deficit",
    "language or speech neurological deficit",
    "post-stroke complication topic",
    "chronic or general non-acute stroke topic",
    "stroke mimic or non-stroke differential diagnosis",
]

LABEL_TO_SIGNAL = {
    "acute ischemic stroke presentation": "is_acute",
    "posterior circulation stroke presentation": "is_posterior",
    "hemorrhagic stroke or intracranial hemorrhage presentation": "is_hemorrhage",
    "motor neurological deficit": "has_motor",
    "language or speech neurological deficit": "has_language",
    "post-stroke complication topic": "asks_complication",
    "chronic or general non-acute stroke topic": "is_chronic_or_general",
    "stroke mimic or non-stroke differential diagnosis": "is_mimic_or_nonstroke",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _count_hits(text: str, terms: Iterable[str]) -> int:
    low = _normalize(text)
    return sum(1 for term in terms if term and term.lower() in low)


def _hit_terms(text: str, terms: Iterable[str], limit: int = 12) -> List[str]:
    low = _normalize(text)
    hits = []
    for term in terms:
        if term and term.lower() in low:
            hits.append(term)
        if len(hits) >= limit:
            break
    return hits


def _extract_time_context(text: str) -> Dict[str, Any]:
    q = _normalize(text)
    ctx: Dict[str, Any] = {
        "has_time": False,
        "duration_minutes": None,
        "is_acute_time": False,
        "is_chronic_time": False,
    }

    if _count_hits(q, CHRONIC_OR_GENERAL_PENALTY_TERMS) > 0:
        ctx["has_time"] = True
        ctx["is_chronic_time"] = True

    patterns = [
        (r"\b(?:for|since|within|over|during|last|past)?\s*(\d+(?:\.\d+)?)\s*(minute|minutes|min|mins)\b", 1),
        (r"\b(?:for|since|within|over|during|last|past)?\s*(\d+(?:\.\d+)?)\s*(hour|hours|hr|hrs)\b", 60),
        (r"\b(?:for|since|within|over|during|last|past)?\s*(\d+(?:\.\d+)?)\s*(day|days)\b", 1440),
        (r"\b(?:for|since|within|over|during|last|past)?\s*(\d+(?:\.\d+)?)\s*(week|weeks|month|months|year|years)\b", None),
    ]

    for pattern, multiplier in patterns:
        m = re.search(pattern, q)
        if not m:
            continue
        ctx["has_time"] = True
        if multiplier is None:
            ctx["is_chronic_time"] = True
            ctx["is_acute_time"] = False
        else:
            minutes = float(m.group(1)) * multiplier
            ctx["duration_minutes"] = minutes
            ctx["is_acute_time"] = minutes <= 24 * 60
            ctx["is_chronic_time"] = minutes >= 14 * 24 * 60
        break

    vague_acute = [
        "sudden onset", "acute onset", "new onset", "abrupt onset",
        "started today", "began today", "developed today", "today",
        "this morning", "this afternoon", "this evening", "last night",
        "few minutes", "several minutes", "few hours", "several hours",
        "last known well", "woke up with", "wake-up",
    ]
    if any(t in q for t in vague_acute):
        ctx["has_time"] = True
        if not ctx["is_chronic_time"]:
            ctx["is_acute_time"] = True

    return ctx


@lru_cache(maxsize=1)
def _signal_cfg() -> Dict[str, Any]:
    return (_load_cfg().get("clinical_signals") or {}) if isinstance(_load_cfg(), dict) else {}


def _model_requested() -> bool:
    env = os.getenv("MEDRAG_USE_SIGNAL_MODEL")
    if env is not None:
        return env.strip().lower() in {"1", "true", "yes", "on"}
    return bool(_signal_cfg().get("model_enabled", False))


_ZERO_SHOT = None
_ZERO_SHOT_ERROR: Optional[str] = None


def _get_zero_shot_pipeline():
    """Lazy optional transformer loader. Never raises to caller."""
    global _ZERO_SHOT, _ZERO_SHOT_ERROR
    if _ZERO_SHOT is not None:
        return _ZERO_SHOT
    if _ZERO_SHOT_ERROR is not None:
        return None

    try:
        from transformers import pipeline  # type: ignore

        model_name = str(
            _signal_cfg().get(
                "model_name",
                os.getenv("MEDRAG_SIGNAL_MODEL", "facebook/bart-large-mnli"),
            )
        )
        device = -1
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                device = 0
        except Exception:
            pass

        _ZERO_SHOT = pipeline("zero-shot-classification", model=model_name, device=device)
        return _ZERO_SHOT
    except Exception as exc:  # model not installed / no internet / incompatible env
        _ZERO_SHOT_ERROR = repr(exc)
        return None


def _score_from_hits(count: int, cap: int = 4) -> float:
    if count <= 0:
        return 0.0
    return min(1.0, float(count) / float(max(1, cap)))


def _rule_signal_scores(text: str) -> Dict[str, Any]:
    low = _normalize(text)
    is_ar = _contains_arabic(text)
    time_ctx = _extract_time_context(low)

    acute_hits = _count_hits(low, ACUTE_HINT_TERMS) + (_count_hits(low, AR_ACUTE_HINTS) if is_ar else 0)
    motor_hits = _count_hits(low, MOTOR_TERMS)
    language_hits = _count_hits(low, LANGUAGE_TERMS)
    posterior_hits = _count_hits(low, POSTERIOR_TERMS) + (_count_hits(low, AR_POSTERIOR_HINTS) if is_ar else 0)
    hemorrhage_hits = _count_hits(low, HEMORRHAGE_TERMS) + (_count_hits(low, AR_HEMORRHAGE_HINTS) if is_ar else 0)
    complication_hits = _count_hits(low, COMPLICATION_TOPICS)
    chronic_hits = _count_hits(low, CHRONIC_OR_GENERAL_PENALTY_TERMS)
    mimic_hits = _count_hits(low, MIMIC_TERMS)

    acute_score = _score_from_hits(acute_hits, 5)
    if time_ctx.get("is_acute_time"):
        acute_score = max(acute_score, 0.85)
    if time_ctx.get("is_chronic_time"):
        acute_score = min(acute_score, 0.25)

    # Focal neuro signs within an acute time window strengthen acute presentation.
    if (motor_hits or language_hits or posterior_hits) and time_ctx.get("is_acute_time"):
        acute_score = max(acute_score, 0.90)

    scores: Dict[str, Any] = {
        "acute_score": acute_score,
        "motor_score": _score_from_hits(motor_hits, 3),
        "language_score": _score_from_hits(language_hits, 3),
        "posterior_score": _score_from_hits(posterior_hits, 3),
        "hemorrhage_score": _score_from_hits(hemorrhage_hits, 3),
        "complication_score": _score_from_hits(complication_hits, 2),
        "chronic_score": max(_score_from_hits(chronic_hits, 3), 1.0 if time_ctx.get("is_chronic_time") else 0.0),
        "mimic_score": _score_from_hits(mimic_hits, 2),
        "time_context": time_ctx,
        "rule_hits": {
            "acute": _hit_terms(low, list(ACUTE_HINT_TERMS) + (AR_ACUTE_HINTS if is_ar else [])),
            "motor": _hit_terms(low, MOTOR_TERMS),
            "language": _hit_terms(low, LANGUAGE_TERMS),
            "posterior": _hit_terms(low, list(POSTERIOR_TERMS) + (AR_POSTERIOR_HINTS if is_ar else [])),
            "hemorrhage": _hit_terms(low, list(HEMORRHAGE_TERMS) + (AR_HEMORRHAGE_HINTS if is_ar else [])),
            "complication": _hit_terms(low, COMPLICATION_TOPICS),
            "chronic": _hit_terms(low, CHRONIC_OR_GENERAL_PENALTY_TERMS),
            "mimic": _hit_terms(low, MIMIC_TERMS),
        },
    }
    return scores


def _model_signal_scores(text: str) -> Dict[str, Any]:
    if not _model_requested():
        return {"model_enabled": False, "used_model": False, "model_scores": {}, "model_error": None}

    clf = _get_zero_shot_pipeline()
    if clf is None:
        return {
            "model_enabled": True,
            "used_model": False,
            "model_scores": {},
            "model_error": _ZERO_SHOT_ERROR,
        }

    try:
        max_chars = int(_signal_cfg().get("max_model_chars", 900))
    except Exception:
        max_chars = 900

    try:
        output = clf((text or "")[:max_chars], MODEL_LABELS, multi_label=True)
        labels = output.get("labels", [])
        scores = output.get("scores", [])
        raw = {str(label): float(score) for label, score in zip(labels, scores)}
        mapped = {LABEL_TO_SIGNAL[label]: score for label, score in raw.items() if label in LABEL_TO_SIGNAL}
        return {"model_enabled": True, "used_model": True, "model_scores": mapped, "model_raw_labels": raw, "model_error": None}
    except Exception as exc:
        return {"model_enabled": True, "used_model": False, "model_scores": {}, "model_error": repr(exc)}


# [تم حذف كود ميت مكرر غير مستخدم: detect_clinical_signals (old v1) — السطر الأصلي 349-413]
def build_dynamic_query_expansions(text: str, signals: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Builds focused expansions instead of blindly appending DEFAULT_STROKE_EXPANSION.
    The original query is not included; returned strings are suffix expansions.
    """
    s = signals or detect_clinical_signals(text, use_model=False)
    expansions: List[str] = []

    if s.get("is_hemorrhage"):
        expansions.append(
            "intracerebral hemorrhage subarachnoid hemorrhage thunderclap headache CT angiography acute intracranial bleeding"
        )

    if s.get("is_posterior") and (s.get("is_acute") or not s.get("is_chronic_or_general")):
        expansions.append(
            "posterior circulation stroke vertebrobasilar brainstem infarction cerebellar infarction vertigo diplopia ataxia MRI DWI CTA"
        )

    if s.get("has_motor") or s.get("has_language"):
        expansions.append(
            "acute ischemic stroke sudden focal neurological deficit hemiparesis aphasia dysarthria facial droop FAST NIHSS thrombolysis thrombectomy"
        )

    if s.get("asks_complication"):
        expansions.append(
            "post-stroke complication depression delirium seizure epilepsy management prognosis"
        )

    if s.get("is_chronic_or_general") and not s.get("is_acute"):
        expansions.append(
            "chronic stroke rehabilitation long-term outcomes vascular cognitive impairment differential diagnosis"
        )

    # Conservative general fallback: only when no focused class was detected.
    if not expansions:
        expansions.append(
            "stroke differential diagnosis clinical presentation emergency evaluation CT MRI evidence"
        )

    if s.get("contains_arabic"):
        expansions.append(
            "acute stroke sudden neurological deficit ischemic stroke hemorrhage posterior circulation"
        )

    # Deduplicate while preserving order.
    return list(dict.fromkeys(e.strip() for e in expansions if e and e.strip()))


def summarize_signals(signals: Dict[str, Any]) -> str:
    """Compact debug/explainability string."""
    keys = [
        "is_acute", "has_motor", "has_language", "is_posterior", "is_hemorrhage",
        "asks_complication", "is_chronic_or_general", "is_mimic_or_nonstroke",
    ]
    active = [k for k in keys if signals.get(k)]
    return ", ".join(active) if active else "no strong clinical signal"


# ============================================================
# v7.12.13 posterior-vomiting hemorrhage guard
# Added as a non-invasive wrapper around detect_clinical_signals.
# Goal:
#   vertigo + ataxia + diplopia + vomiting should remain posterior-pattern,
#   not hemorrhage-pattern, unless hard hemorrhage cues exist.
# ============================================================

_V71213_POSTERIOR_CUES = [
    "vertigo", "dizziness", "ataxia", "gait ataxia", "gait difficulty",
    "difficulty walking", "unsteady gait", "diplopia", "double vision",
    "nystagmus", "dysmetria", "brainstem", "cerebellar",
    "posterior circulation", "vertebrobasilar", "basilar",
    "دوار", "دوخة", "ترنح", "ازدواج", "صعوبة المشي", "رأرأة", "جذع الدماغ",
]

_V71213_HARD_HEMORRHAGE_CUES = [
    "thunderclap", "worst headache", "worst headache of life",
    "sudden severe headache", "severe headache",
    "neck stiffness", "meningismus", "photophobia",
    "decreased consciousness", "loss of consciousness", "reduced consciousness",
    "altered consciousness", "coma", "seizure",
    "intraparenchymal", "intracerebral hemorrhage", "subarachnoid hemorrhage",
    "hemorrhagic stroke", "hematoma", "parenchymal blood",
    "ct shows hemorrhage", "ct showed hemorrhage", "brain bleed",
    "intracranial bleeding", "hemorrhage on ct",
    "hypertensive emergency", "hypertensive crisis", "malignant hypertension",
    "very high blood pressure", "severe hypertension",
    "warfarin", "anticoagulation", "anticoagulant",
    "rivaroxaban", "apixaban", "dabigatran",
    "صداع شديد", "صداع مفاجئ", "أسوأ صداع", "فقدان وعي", "تدهور وعي",
    "نزف", "نزيف", "ورم دموي",
]

_V71213_VOMITING_CUES = [
    "vomiting", "emesis", "nausea and vomiting",
    "قيء", "إقياء", "استفراغ",
]


def _v71213_has_any(text: str, terms: list[str]) -> bool:
    low = (text or "").lower()
    return any(t in low for t in terms)


def _v71213_count_hits(text: str, terms: list[str]) -> int:
    low = (text or "").lower()
    return sum(1 for t in terms if t in low)


def _v71213_posterior_without_hard_hemorrhage(text: str) -> bool:
    low = (text or "").lower()
    posterior_hits = _v71213_count_hits(low, _V71213_POSTERIOR_CUES)
    hard_heme = _v71213_has_any(low, _V71213_HARD_HEMORRHAGE_CUES)
    vomiting = _v71213_has_any(low, _V71213_VOMITING_CUES)

    return bool(posterior_hits >= 2 and vomiting and not hard_heme)


if "_v71213_original_detect_clinical_signals" not in globals():
    _v71213_original_detect_clinical_signals = detect_clinical_signals

    def detect_clinical_signals(text: str, use_model=None):
        s = _v71213_original_detect_clinical_signals(text, use_model=use_model)

        if _v71213_posterior_without_hard_hemorrhage(text):
            s["is_hemorrhage"] = False

            scores = s.get("scores")
            if isinstance(scores, dict):
                scores["hemorrhage"] = 0.0

            rule_hits = s.get("rule_hits")
            if isinstance(rule_hits, dict):
                old_hits = rule_hits.get("hemorrhage", []) or []
                rule_hits["hemorrhage"] = [
                    h for h in old_hits
                    if str(h).lower() not in {"vomiting", "قيء", "إقياء", "استفراغ"}
                ]

        return s


# ============================================================
# v7.15 clinical safety override: shared negation/temporal logic
# ============================================================
# The legacy detector remains available for compatibility and optional model
# metadata, but final high-stakes clinical flags are reconciled against the
# shared context parser.  This prevents negated terms from activating query
# expansion or Evidence Judge recall gates.
from app.rag.clinical_context import analyze_case_context as _v715_analyze_case_context

if "_v715_original_detect_clinical_signals" not in globals():
    _v715_original_detect_clinical_signals = detect_clinical_signals


def detect_clinical_signals(text: str, use_model: Optional[bool] = None) -> Dict[str, Any]:
    legacy = _v715_original_detect_clinical_signals(text, use_model=use_model)
    context = _v715_analyze_case_context(text or "")
    positives = context.get("positive_findings") or {}
    negatives = context.get("negative_findings") or {}

    def ratio(items: Any, cap: int = 3) -> float:
        count = len(list(items or []))
        return min(1.0, count / float(max(cap, 1))) if count else 0.0

    scores = {
        "acute": 0.90 if context.get("acute_onset") and context.get("focal_neuro") else (0.70 if context.get("acute_onset") else 0.0),
        "motor": ratio(positives.get("motor"), 3),
        "language": ratio(positives.get("language"), 3),
        "posterior": ratio(positives.get("posterior"), 3),
        "hemorrhage": 1.0 if context.get("hemorrhage_warning") else 0.0,
        "complication": 1.0 if context.get("complication_present") else 0.0,
        "chronic": 1.0 if context.get("chronic_context") else 0.0,
        "mimic": 1.0 if context.get("mimic_present") else 0.0,
    }

    return {
        "is_acute": bool(context.get("acute_onset")),
        "has_motor": bool(context.get("motor_deficit")),
        "has_language": bool(context.get("language_deficit")),
        "is_posterior": bool(context.get("posterior_pattern")),
        "is_hemorrhage": bool(context.get("hemorrhage_warning")),
        "asks_complication": bool(context.get("complication_present")),
        "is_chronic_or_general": bool(context.get("chronic_context")),
        "is_mimic_or_nonstroke": bool(context.get("mimic_present")),
        "contains_arabic": _contains_arabic(text or ""),
        "source": "clinical_context_v4+optional_transformer_metadata",
        "scores": scores,
        "time_context": dict(context.get("time_context") or {}),
        "rule_hits": {
            "acute": list(positives.get("acute") or []),
            "motor": list(positives.get("motor") or []),
            "language": list(positives.get("language") or []),
            "posterior": list(positives.get("posterior") or []),
            "hemorrhage": list(positives.get("hemorrhage") or []) if context.get("hemorrhage_warning") else [],
            "complication": list(positives.get("hemorrhage") or []) if context.get("complication_present") else [],
            "chronic": [],
            "mimic": list(positives.get("mimic") or []),
            "persistent": list(positives.get("persistent") or []),
            "resolved": list(positives.get("resolved") or []),
        },
        "negated_rule_hits": {
            "hemorrhage": list(negatives.get("hemorrhage") or []),
        },
        "persistent_deficit": bool(context.get("persistent_deficit")),
        "transient_resolved_episode": bool(context.get("transient_resolved_episode")),
        "context_profile": context,
        # Keep optional model diagnostics, but do not let a model override an
        # explicit local negation in the case text.
        "model": dict(legacy.get("model") or {}),
    }
