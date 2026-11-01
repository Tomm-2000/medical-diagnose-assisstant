# -*- coding: utf-8 -*-
"""Shared deterministic clinical-context parsing for MedRAG.

This module is deliberately conservative.  It does not diagnose a patient.
It normalizes case-text facts used by routing, retrieval, Evidence Judge and
explainability so all layers agree about:

* affirmed versus negated findings;
* ongoing/persistent versus completely resolved deficits;
* acute focal neurological presentation;
* positive hemorrhage warning features;
* time context and documented risk factors.

The parser implements local negation scope with clause boundaries.  A finding
mentioned after ``no``, ``denies``, ``without`` (and Arabic equivalents) is not
allowed to activate a positive clinical flag.  Short symptom duration alone is
never treated as TIA; explicit complete resolution is required.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import re

_AR_RE = re.compile(r"[\u0600-\u06FF]")
_DURATION_RE = re.compile(
    r"\b(?:for|since|within|over|during|last|past|after)?\s*"
    r"(\d+(?:\.\d+)?)\s*"
    r"(minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\b",
    re.I,
)

# Negation terms are intentionally short and local.  The clause/window logic
# prevents a negation in one sentence from leaking into another finding.
NEGATION_PREFIXES: Tuple[str, ...] = (
    "no", "not", "without", "denies", "deny", "denied", "negative for",
    "absence of", "absent", "free of", "rules out", "ruled out",
    "لا", "ليس", "ليست", "بدون", "ينفي", "نفت", "لا يوجد", "لا توجد",
    "غياب", "غير موجود", "غير موجودة",
)
NEGATION_SUFFIXES: Tuple[str, ...] = (
    "is absent", "are absent", "was absent", "were absent",
    "is not present", "are not present", "was not present", "were not present",
    "غير موجود", "غير موجودة", "منفي", "منفية",
)

# Contrast terms terminate the practical scope of a preceding negation.
_CLAUSE_BOUNDARY_RE = re.compile(
    r"[\n\r.!?;:]|\bbut\b|\bhowever\b|\balthough\b|\bexcept\b|"
    r"\byet\b|\bnevertheless\b|\bلكن\b|\bولكن\b|\bإلا\b",
    re.I,
)
_WORD_RE = re.compile(r"[a-z0-9]+|[\u0600-\u06FF]+", re.I)

ACUTE_CUES = (
    "sudden", "suddenly", "acute", "abrupt", "new onset", "developed",
    "started", "began", "last known well", "wake-up", "woke up",
    "مفاجئ", "فجأة", "حاد", "بدأ", "ظهرت",
)
MOTOR_CUES = (
    "right-sided facial and arm weakness", "left-sided facial and arm weakness",
    "right-sided weakness", "left-sided weakness", "unilateral weakness",
    "facial weakness", "facial droop", "arm weakness", "leg weakness",
    "weakness", "hemiparesis", "hemiplegia", "arm drift",
    "ضعف نصفي", "شلل نصفي", "ضعف الوجه", "ميلان الوجه", "ضعف الذراع",
    "ضعف الساق", "ضعف", "شلل",
)
LANGUAGE_CUES = (
    "expressive aphasia", "global aphasia", "aphasia", "dysarthria",
    "slurred speech", "speech difficulty", "speech disturbance",
    "حبسة", "تلعثم", "ثقل الكلام", "صعوبة الكلام",
)
POSTERIOR_CUES = (
    "vertigo", "diplopia", "double vision", "ataxia", "gait ataxia",
    "truncal ataxia", "nystagmus", "dysmetria", "dysphagia", "hoarseness",
    "brainstem", "cerebellar", "posterior circulation", "vertebrobasilar",
    "دوار", "ازدواج الرؤية", "ترنح", "رأرأة", "عسر البلع", "جذع الدماغ",
)

# Positive SAH/ICH indicators.  Hypertension or atrial fibrillation alone are
# never hemorrhage-warning cues.
SAH_CORE_CUES = (
    "thunderclap headache", "thunderclap", "worst headache of life",
    "worst-ever headache", "worst ever headache", "explosive headache",
    "sudden severe headache", "abrupt severe headache",
    "صداع انفجاري", "أسوأ صداع", "صداع شديد مفاجئ",
)
SAH_CONTEXT_CUES = (
    "neck stiffness", "stiff neck", "meningismus", "photophobia",
    "collapse", "syncope", "loss of consciousness", "decreased consciousness",
    "seizure", "vomiting", "ruptured aneurysm", "basal cistern blood",
    "تيبس الرقبة", "رهاب الضوء", "فقدان الوعي", "اختلاج", "قيء",
)
EXPLICIT_HEMORRHAGE_CUES = (
    "ct shows hemorrhage", "ct showed hemorrhage", "hemorrhage on ct",
    "intracerebral hemorrhage", "intracerebral haemorrhage",
    "intraparenchymal hemorrhage", "intraparenchymal haemorrhage",
    "subarachnoid hemorrhage", "subarachnoid haemorrhage",
    "parenchymal hematoma", "intraparenchymal hematoma", "brain bleed",
    "intracranial bleeding", "basal ganglia hematoma", "subarachnoid blood",
    "نزف داخل الدماغ", "نزف تحت العنكبوتية", "نزيف داخل القحف", "ورم دموي",
)
ALTERED_CONSCIOUSNESS_CUES = (
    "loss of consciousness", "decreased consciousness", "reduced consciousness",
    "altered consciousness", "coma", "drowsiness", "rapid neurological decline",
    "فقدان الوعي", "تدهور الوعي", "نقص الوعي", "غيبوبة",
)
SEIZURE_CUES = ("seizure", "seizures", "convulsion", "اختلاج", "نوبة صرعية")
VOMITING_CUES = ("vomiting", "repeated vomiting", "emesis", "قيء", "إقياء", "استفراغ")

PERSISTENCE_CUES = (
    "neurological deficits are still present", "neurologic deficits are still present",
    "deficits are still present", "symptoms are still present", "still present",
    "persistent deficit", "persistent deficits", "persistent weakness",
    "persistent aphasia", "persistent dysarthria", "ongoing deficit",
    "ongoing symptoms", "symptoms persist", "deficits persist", "continues",
    "continuing", "not resolved", "has not resolved", "have not resolved",
    "residual deficit", "remains weak", "remain present",
    "الأعراض ما زالت موجودة", "الأعراض مستمرة", "العجز مستمر", "لم تتحسن",
    "لم تختف", "ما زال الضعف", "مستمر", "مستمرة",
)
RESOLUTION_CUES = (
    "completely resolved", "fully resolved", "resolved completely",
    "complete recovery", "full recovery", "back to baseline",
    "returned to baseline", "symptoms gone", "deficits resolved",
    "no residual deficit", "no persistent deficit", "asymptomatic on arrival",
    "normal neurological examination", "normal neurologic examination",
    "الأعراض زالت تماماً", "الأعراض زالت تماما", "اختفت الأعراض", "عاد لطبيعته",
    "لا يوجد عجز متبق", "تحسن كامل",
)

CHRONIC_CUES = (
    "chronic", "gradual", "progressive", "for months", "for years",
    "several months", "several years", "مزمن", "تدريجي", "منذ أشهر", "منذ سنوات",
)
MIMIC_CUES = (
    "hypoglycemia", "hypoglycaemia", "postictal", "migraine aura",
    "functional neurological", "conversion disorder", "bppv", "positional vertigo",
    "نقص سكر", "بعد الاختلاج", "شقيقة مع هالة", "اضطراب وظيفي",
)

RISK_FACTOR_TERMS: Dict[str, Tuple[str, ...]] = {
    "hypertension": ("hypertension", "high blood pressure", "ارتفاع ضغط الدم", "ضغط"),
    "atrial_fibrillation": ("atrial fibrillation", "afib", "a-fib", "رجفان أذيني"),
    "anticoagulant_use": (
        "warfarin", "apixaban", "rivaroxaban", "dabigatran", "edoxaban",
        "anticoagulant", "anticoagulation", "مميع", "وارفارين",
    ),
    "diabetes": ("diabetes", "diabetic", "داء السكري", "سكري"),
}


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    p = normalize_text(phrase)
    if re.fullmatch(r"[a-z0-9 ]+", p):
        return re.compile(r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])", re.I)
    return re.compile(re.escape(p), re.I)


def _clause_window(text: str, start: int, end: int) -> Tuple[str, str]:
    left_region = text[:start]
    right_region = text[end:]
    left_matches = list(_CLAUSE_BOUNDARY_RE.finditer(left_region))
    left_bound = left_matches[-1].end() if left_matches else 0
    right_match = _CLAUSE_BOUNDARY_RE.search(right_region)
    right_bound = end + (right_match.start() if right_match else len(right_region))
    return text[left_bound:start], text[end:right_bound]


def phrase_mentions(text: str, phrase: str, token_window: int = 8) -> List[Dict[str, Any]]:
    """Return every phrase mention with local affirmed/negated status."""
    low = normalize_text(text)
    if not low or not phrase:
        return []
    mentions: List[Dict[str, Any]] = []
    for match in _phrase_pattern(phrase).finditer(low):
        left_clause, right_clause = _clause_window(low, match.start(), match.end())
        left_tokens = _WORD_RE.findall(left_clause)[-token_window:]
        left_local = " ".join(left_tokens)
        right_tokens = _WORD_RE.findall(right_clause)[:5]
        right_local = " ".join(right_tokens)

        negated = False
        for neg in NEGATION_PREFIXES:
            n = normalize_text(neg)
            if n and re.search(r"(?:^|\s)" + re.escape(n) + r"(?:\s|$)", left_local):
                negated = True
                break
        if not negated:
            for suffix in NEGATION_SUFFIXES:
                if normalize_text(suffix) in right_local:
                    negated = True
                    break

        mentions.append({
            "phrase": phrase,
            "start": match.start(),
            "end": match.end(),
            "negated": negated,
            "status": "negated" if negated else "affirmed",
            "left_context": left_clause[-100:],
            "right_context": right_clause[:100],
        })
    return mentions


def phrase_is_affirmed(text: str, phrase: str) -> bool:
    mentions = phrase_mentions(text, phrase)
    return any(not m["negated"] for m in mentions)


def phrase_is_negated(text: str, phrase: str) -> bool:
    mentions = phrase_mentions(text, phrase)
    return bool(mentions) and all(m["negated"] for m in mentions)


def affirmed_terms(text: str, terms: Iterable[str]) -> List[str]:
    return [term for term in terms if phrase_is_affirmed(text, term)]


def negated_terms(text: str, terms: Iterable[str]) -> List[str]:
    return [term for term in terms if phrase_is_negated(text, term)]


def _extract_time_context(text: str) -> Dict[str, Any]:
    low = normalize_text(text)
    result: Dict[str, Any] = {
        "has_time": False,
        "duration_minutes": None,
        "is_acute_time": False,
        "is_chronic_time": False,
        "time_phrase": None,
    }
    match = _DURATION_RE.search(low)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        multiplier = 1.0
        if unit.startswith(("hour", "hr")):
            multiplier = 60.0
        elif unit.startswith("day"):
            multiplier = 1440.0
        elif unit.startswith("week"):
            multiplier = 10080.0
        elif unit.startswith("month"):
            multiplier = 43200.0
        elif unit.startswith("year"):
            multiplier = 525600.0
        minutes = value * multiplier
        result.update({
            "has_time": True,
            "duration_minutes": minutes,
            "is_acute_time": minutes <= 1440.0,
            "is_chronic_time": minutes >= 20160.0,
            "time_phrase": match.group(0),
        })
    elif affirmed_terms(low, ACUTE_CUES):
        result.update({"has_time": True, "is_acute_time": True, "time_phrase": "acute_onset_cue"})
    return result


def _first_span(text: str, terms: Sequence[str]) -> Dict[str, Any] | None:
    low = normalize_text(text)
    for term in sorted(terms, key=len, reverse=True):
        for mention in phrase_mentions(low, term):
            if not mention["negated"]:
                start, end = mention["start"], mention["end"]
                return {
                    "matched_cue": term,
                    "start": start,
                    "end": end,
                    "text_span": low[max(0, start - 55):min(len(low), end + 55)],
                }
    return None


@dataclass
class ClinicalContext:
    acute_onset: bool
    focal_neuro: bool
    motor_deficit: bool
    language_deficit: bool
    posterior_pattern: bool
    persistent_deficit: bool
    explicit_resolution: bool
    transient_resolved_episode: bool
    hemorrhage_warning: bool
    sah_warning: bool
    ich_warning: bool
    complication_present: bool
    mimic_present: bool
    chronic_context: bool
    positive_findings: Dict[str, List[str]]
    negative_findings: Dict[str, List[str]]
    risk_factors: Dict[str, bool]
    time_context: Dict[str, Any]
    source: str = "clinical_context_v4_negation_temporal"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyze_case_context(case_text: str) -> Dict[str, Any]:
    """Build a conservative, shared clinical-context profile."""
    text = str(case_text or "")
    low = normalize_text(text)

    acute_hits = affirmed_terms(low, ACUTE_CUES)
    motor_hits = affirmed_terms(low, MOTOR_CUES)
    language_hits = affirmed_terms(low, LANGUAGE_CUES)
    posterior_hits = affirmed_terms(low, POSTERIOR_CUES)
    persistence_hits = [cue for cue in PERSISTENCE_CUES if cue in low]
    resolution_hits = affirmed_terms(low, RESOLUTION_CUES)

    # Phrases such as "not resolved" are positive persistence evidence, whereas
    # a locally negated "resolved" is not complete resolution.
    persistent = bool(persistence_hits)
    explicit_resolution = bool(resolution_hits) and not persistent

    focal = bool(motor_hits or language_hits or posterior_hits)
    time_context = _extract_time_context(low)
    acute_onset = bool(acute_hits or (focal and time_context.get("is_acute_time")))

    explicit_heme = affirmed_terms(low, EXPLICIT_HEMORRHAGE_CUES)
    sah_core = affirmed_terms(low, SAH_CORE_CUES)
    sah_context = affirmed_terms(low, SAH_CONTEXT_CUES)
    altered = affirmed_terms(low, ALTERED_CONSCIOUSNESS_CUES)
    seizure = affirmed_terms(low, SEIZURE_CUES)
    vomiting = affirmed_terms(low, VOMITING_CUES)

    # SAH warning requires a strong headache core plus context, or explicit
    # SAH/aneurysmal/imaging evidence.  Generic focal deficits do not satisfy it.
    explicit_sah = [x for x in explicit_heme if "subarachnoid" in x or "basal cistern" in x]
    sah_warning = bool(explicit_sah or (sah_core and sah_context))

    # ICH warning requires direct bleeding/imaging language, or a clinically
    # hemorrhage-weighted presentation with impaired consciousness/seizure and
    # severe headache/vomiting.  Hypertension alone is insufficient.
    explicit_ich = [x for x in explicit_heme if any(k in x for k in (
        "intracerebral", "intraparenchymal", "hematoma", "brain bleed", "intracranial", "نزف داخل"
    ))]
    severe_headache = affirmed_terms(low, ("severe headache", "sudden severe headache", "worst headache of life", "صداع شديد"))
    ich_warning = bool(explicit_ich or ((altered or seizure) and (severe_headache or vomiting)))
    hemorrhage_warning = bool(sah_warning or ich_warning or explicit_heme)

    complication_present = bool(seizure)
    mimic_hits = affirmed_terms(low, MIMIC_CUES)
    chronic_hits = affirmed_terms(low, CHRONIC_CUES)

    all_heme_terms = tuple(dict.fromkeys(
        EXPLICIT_HEMORRHAGE_CUES + SAH_CORE_CUES + SAH_CONTEXT_CUES
        + ALTERED_CONSCIOUSNESS_CUES + SEIZURE_CUES + VOMITING_CUES
    ))
    neg_heme = negated_terms(low, all_heme_terms)

    risk_factors = {
        key: bool(affirmed_terms(low, terms))
        for key, terms in RISK_FACTOR_TERMS.items()
    }

    return ClinicalContext(
        acute_onset=acute_onset,
        focal_neuro=focal,
        motor_deficit=bool(motor_hits),
        language_deficit=bool(language_hits),
        posterior_pattern=bool(posterior_hits),
        persistent_deficit=persistent,
        explicit_resolution=explicit_resolution,
        transient_resolved_episode=bool(focal and explicit_resolution and not persistent),
        hemorrhage_warning=hemorrhage_warning,
        sah_warning=sah_warning,
        ich_warning=ich_warning,
        complication_present=complication_present,
        mimic_present=bool(mimic_hits),
        chronic_context=bool(chronic_hits or time_context.get("is_chronic_time")),
        positive_findings={
            "acute": acute_hits,
            "motor": motor_hits,
            "language": language_hits,
            "posterior": posterior_hits,
            "persistent": persistence_hits,
            "resolved": resolution_hits,
            "hemorrhage": list(dict.fromkeys(explicit_heme + sah_core + sah_context + altered + seizure + vomiting)),
            "mimic": mimic_hits,
        },
        negative_findings={
            "hemorrhage": neg_heme,
        },
        risk_factors=risk_factors,
        time_context=time_context,
    ).to_dict()


def clinical_fact_spans(case_text: str) -> List[Dict[str, Any]]:
    """Return semantic patient facts for SourceCheckup claim generation."""
    profile = analyze_case_context(case_text)
    facts: List[Dict[str, Any]] = []

    def add_fact(fact_key: str, claim_text: str, terms: Sequence[str], polarity: str = "positive") -> None:
        span = _first_span(case_text, terms)
        if span is None and polarity == "negative":
            # For negative facts use the first locally negated mention.
            low = normalize_text(case_text)
            for term in sorted(terms, key=len, reverse=True):
                mentions = phrase_mentions(low, term)
                neg = next((m for m in mentions if m["negated"]), None)
                if neg:
                    span = {
                        "matched_cue": term,
                        "start": neg["start"],
                        "end": neg["end"],
                        "text_span": low[max(0, neg["start"] - 55):min(len(low), neg["end"] + 55)],
                    }
                    break
        if span is not None:
            facts.append({
                "fact_key": fact_key,
                "claim_text": claim_text,
                "polarity": polarity,
                "evidence": {"source": "patient_case", **span},
            })

    if profile["acute_onset"]:
        add_fact("acute_onset", "The neurological symptoms had an acute or sudden onset.", ACUTE_CUES)
    if profile["persistent_deficit"]:
        add_fact("persistent_deficit", "The neurological deficits are still present.", PERSISTENCE_CUES)
    elif profile["explicit_resolution"]:
        add_fact("resolved_deficit", "The neurological deficits completely resolved.", RESOLUTION_CUES)

    if profile["motor_deficit"]:
        add_fact("motor_deficit", "The patient has a focal motor deficit.", MOTOR_CUES)
    if profile["language_deficit"]:
        add_fact("language_deficit", "The patient has aphasia or dysarthria/slurred speech.", LANGUAGE_CUES)
    if profile["posterior_pattern"]:
        add_fact("posterior_pattern", "The case contains posterior-circulation-compatible findings.", POSTERIOR_CUES)

    negative_map = {
        "thunderclap": ("The patient denies thunderclap or sudden severe headache.", SAH_CORE_CUES),
        "seizure": ("The patient denies seizure.", SEIZURE_CUES),
        "loss_of_consciousness": ("The patient denies loss or reduction of consciousness.", ALTERED_CONSCIOUSNESS_CUES),
        "vomiting": ("The patient denies vomiting.", VOMITING_CUES),
    }
    for key, (claim, terms) in negative_map.items():
        if negated_terms(case_text, terms):
            add_fact(key, claim, terms, polarity="negative")

    if profile["risk_factors"].get("hypertension"):
        add_fact("hypertension", "The patient has hypertension.", RISK_FACTOR_TERMS["hypertension"])
    if profile["risk_factors"].get("atrial_fibrillation"):
        add_fact("atrial_fibrillation", "The patient has atrial fibrillation.", RISK_FACTOR_TERMS["atrial_fibrillation"])
    if profile["risk_factors"].get("anticoagulant_use"):
        add_fact("anticoagulant_use", "The patient is documented to use an anticoagulant.", RISK_FACTOR_TERMS["anticoagulant_use"])

    # Add a compact exact time claim without turning words such as "minute" into
    # standalone clinical claims.
    duration = profile["time_context"].get("duration_minutes")
    phrase = profile["time_context"].get("time_phrase")
    if duration is not None and phrase:
        match = re.search(re.escape(str(phrase)), normalize_text(case_text))
        if match:
            facts.append({
                "fact_key": "time_from_onset",
                "claim_text": f"The documented symptom time is approximately {duration:g} minutes.",
                "polarity": "positive",
                "evidence": {
                    "source": "patient_case",
                    "matched_cue": phrase,
                    "start": match.start(),
                    "end": match.end(),
                    "text_span": normalize_text(case_text)[max(0, match.start()-45):min(len(normalize_text(case_text)), match.end()+45)],
                },
            })

    # De-duplicate by semantic fact key.
    out: List[Dict[str, Any]] = []
    seen = set()
    for fact in facts:
        if fact["fact_key"] in seen:
            continue
        seen.add(fact["fact_key"])
        out.append(fact)
    return out
