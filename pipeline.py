# app/rag/pipeline.py
from __future__ import annotations

from typing import List, Dict, Optional
import os
import re
import yaml
import numpy as np

from app.rag.utils_text import safe_truncate
from app.models.medcpt_reranker import rerank_docs
from app.models.llm import generate_answer
from app.rag.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

_AR_RE = re.compile(r"[\u0600-\u06FF]")


def _contains_arabic(text: str) -> bool:
    return bool(_AR_RE.search(text or ""))


def _translate_if_arabic(text: str) -> str:
    return text


PIPELINE_VERSION = "v7.7-kg-pipeline-final"


try:
    from app.models.stroke_risk import predict_stroke_risk
    _HAS_STROKE_MODEL = True
except Exception:
    predict_stroke_risk = None
    _HAS_STROKE_MODEL = False

try:
    from app.rag.retriever import hybrid_search
    _HAS_LOCAL = True
except Exception:
    _HAS_LOCAL = False

with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f) or {}

KG_CFG = CFG.get("knowledge_graph", {})
USE_KG = bool(KG_CFG.get("enabled", False))

try:
    import torch
    from sentence_transformers import SentenceTransformer as _ST

    _ALLOW_CPU = os.getenv("MEDRAG_ALLOW_CPU", "0").strip() == "1"
    if (not torch.cuda.is_available()) and (not _ALLOW_CPU):
        raise RuntimeError(
            "CUDA غير متاح – هذا المشروع مضبوط ليعمل على GPU فقط. "
            "إذا تريد CPU مؤقتاً: set MEDRAG_ALLOW_CPU=1"
        )

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    EMB = _ST(
        (CFG.get("models") or {}).get("embeddings", "sentence-transformers/all-MiniLM-L6-v2"),
        device=_device
    )
except Exception:
    EMB = None


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = float(np.linalg.norm(a) + 1e-12)
    nb = float(np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / (na * nb))


# =========================
# RULE ENGINE
# =========================

_NEURO_FOCAL = [
    "weakness", "unilateral weakness", "arm weakness", "leg weakness",
    "hemiparesis", "hemiplegia",
    "aphasia", "expressive aphasia", "slurred speech", "dysarthria",
    "facial droop", "vision loss", "visual loss", "hemianopia",
    "numbness", "paresthesia",
]

_HEM_PATTERN = [
    "thunderclap", "worst headache", "sudden severe headache",
    "repeated vomiting", "vomiting",
    "decreased consciousness", "worsening consciousness", "loss of consciousness",
    "meningismus", "neck stiffness", "photophobia",
    "seizure",
]

_HEM_KEYWORDS = [
    "hemorrhage", "haemorrhage", "bleeding",
    "intracerebral hemorrhage", "subarachnoid hemorrhage",
    "sah", "ich"
]

_ISCH_KEYWORDS = [
    "ischemic", "ischaemic", "infarction", "cerebral infarction",
    "ais", "thrombectomy", "thrombolysis", "tpa", "alteplase"
]

_TIA_KEYWORDS = [
    "tia", "transient ischemic attack", "transient ischaemic attack"
]

_RESOLUTION_TERMS = [
    "resolve", "resolved", "completely resolved", "now back to baseline", "back to baseline",
    "symptoms resolved", "returned to baseline", "return to baseline"
]

_VERTEBROBASILAR_SYMPTOMS = [
    "vertigo", "dizziness", "double vision", "diplopia",
    "ataxia", "gait difficulty", "balance problem",
    "cross sensory", "alternating", "hoarseness", "dysphagia"
]

_FRONTAL_LOBE_SYMPTOMS = [
    "personality change", "apathy", "disinhibition", "broca aphasia",
    "contralateral weakness", "gaze preference"
]

_TEMPORAL_LOBE_SYMPTOMS = [
    "memory loss", "wernicke aphasia", "auditory hallucination",
    "visual field deficit"
]

_OCCIPITAL_LOBE_SYMPTOMS = [
    "homonymous hemianopia", "visual agnosia", "cortical blindness"
]

_CEREBELLAR_SYMPTOMS = [
    "ataxia", "dysmetria", "intention tremor", "nystagmus",
    "slurred speech", "vertigo"
]

_BRAINSTEM_SYMPTOMS = [
    "diplopia", "dysphagia", "dysarthria", "crossed findings",
    "locked-in syndrome", "vertigo", "hearing loss"
]

_MIGRAINE_AURA_SYMPTOMS = [
    "visual aura", "scintillating scotoma", "fortification spectra",
    "unilateral sensory symptoms", "speech disturbance", "followed by headache"
]

_SEIZURE_SYMPTOMS = [
    "tonic-clonic", "jerking", "staring spell", "automatism",
    "postictal confusion", "aura"
]

_DUR_MIN_RE = re.compile(r"(\d{1,3})\s*(minutes|minute|min)\b", re.IGNORECASE)
_DUR_HR_RE = re.compile(r"(\d{1,3})\s*(hours|hour|hr|hrs)\b", re.IGNORECASE)


def _extract_duration_minutes(case: str) -> Optional[int]:
    t = case or ""

    m = _DUR_MIN_RE.search(t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None

    h = _DUR_HR_RE.search(t)
    if h:
        try:
            return int(h.group(1)) * 60
        except Exception:
            return None

    return None


def _has_any(text: str, keys: List[str]) -> bool:
    tt = (text or "").lower()
    return any(k.lower() in tt for k in keys)


def _rule_diagnose_from_case_v2(case_text: str) -> Optional[Dict[str, str]]:
    t = (case_text or "").lower()
    dur_min = _extract_duration_minutes(case_text)
    ongoing = ("ongoing" in t) or ("still" in t) or ("persist" in t) or ("persistent" in t)
    resolved = _has_any(t, _RESOLUTION_TERMS)

    # حماية بسيطة من false positive:
    # isolated vertigo قصير ومحلول لا يجب أن يتحول مباشرة إلى brainstem stroke.
    if (
        resolved
        and dur_min is not None
        and dur_min <= 10
        and ("isolated vertigo" in t or (("vertigo" in t or "dizziness" in t) and not _has_any(t, _NEURO_FOCAL)))
        and not _has_any(t, ["diplopia", "double vision", "ataxia", "gait difficulty", "dysarthria", "dysphagia"])
    ):
        return None

    if _has_any(t, _BRAINSTEM_SYMPTOMS) and (_has_any(t, _VERTEBROBASILAR_SYMPTOMS) or ongoing):
        return {
            "dx": "سكتة في جذع الدماغ محتملة",
            "rationale": "أعراض مثل ازدواج الرؤية، صعوبة البلع، والدوار تشير إلى إصابة جذع الدماغ.",
            "safety": "حالة إسعافية: قد تؤدي إلى متلازمة الانغلاق أو توقف التنفس."
        }

    if _has_any(t, _CEREBELLAR_SYMPTOMS) and ongoing:
        return {
            "dx": "سكتة في المخيخ محتملة",
            "rationale": "أعراض مثل الترنح، الرعاش المقصي، والدوار تشير إلى إصابة المخيخ.",
            "safety": "حالة إسعافية: قد تسبب وذمة دماغية وانفتاق."
        }

    if _has_any(t, _MIGRAINE_AURA_SYMPTOMS) and dur_min and dur_min < 60 and not ongoing:
        return {
            "dx": "صداع نصفي مع أورة محتمل",
            "rationale": "أعراض بؤرية عابرة تسبق الصداع وتستمر أقل من ساعة.",
            "safety": "إذا كانت الأعراض جديدة أو شديدة، استبعد السكتة أولاً."
        }

    if _has_any(t, _SEIZURE_SYMPTOMS) and not ongoing and ("postictal" in t or "confusion" in t):
        return {
            "dx": "نوبة صرع محتملة",
            "rationale": "وجود تشنجات أو تغير في الوعي يتبعه ارتباك.",
            "safety": "قد تكون النوبة أول عرض لمرض عصبي كامن."
        }

    risk_factors = []
    if "hypertension" in t or "htn" in t:
        risk_factors.append("hypertension")
    if "diabetes" in t or "dm" in t:
        risk_factors.append("diabetes")
    if "afib" in t or "atrial fibrillation" in t:
        risk_factors.append("atrial fibrillation")
    if "smoking" in t or "smoker" in t:
        risk_factors.append("smoking")

    if risk_factors and _has_any(t, _NEURO_FOCAL) and ongoing:
        return {
            "dx": "سكتة إقفارية حادة على الأرجح",
            "rationale": f"وجود أعراض بؤرية مستمرة مع عوامل خطر: {', '.join(risk_factors)}.",
            "safety": "تقييم عاجل ضروري لتحديد إمكانية العلاج بمذيبات الجلطات."
        }

    return None


def _rule_diagnose_from_case(case_text: str) -> Optional[Dict[str, str]]:
    t = (case_text or "").lower()
    dur_min = _extract_duration_minutes(case_text)

    has_focal = _has_any(t, _NEURO_FOCAL)
    resolved = _has_any(t, _RESOLUTION_TERMS)
    ongoing = ("ongoing" in t) or ("still" in t) or ("persist" in t) or ("persistent" in t)

    hem_keyword = _has_any(t, _HEM_KEYWORDS)
    hem_pattern = _has_any(t, _HEM_PATTERN)
    severe_htn = ("severe hypertension" in t) or ("hypertension" in t and ("severe" in t or "very high" in t))
    rapid_loc = ("rapid" in t and ("consciousness" in t or "loc" in t)) or ("worsening consciousness" in t)

    if has_focal and resolved and (dur_min is None or dur_min <= 60):
        return {
            "dx": "نوبة إقفارية عابرة (TIA) مرجّحة (أعراض بؤرية مفاجئة اختفت بالكامل خلال فترة قصيرة)",
            "rationale": "الأعراض العصبية البؤرية ظهرت بشكل مفاجئ ثم اختفت كلياً وعاد المريض لخط الأساس خلال مدة قصيرة.",
            "safety": "TIA إنذار لسكتة قريبة: يلزم تقييم إسعافي/سريع وعلاج عوامل الخطورة لمنع السكتة."
        }

    thunder = ("thunderclap" in t) or ("worst headache" in t) or ("sudden severe headache" in t)
    vomiting = "vomiting" in t

    if hem_keyword or ((thunder and (vomiting or rapid_loc)) and (severe_htn or hem_pattern)):
        return {
            "dx": "نزف داخل القحف/نزف دماغي أو نزف تحت العنكبوتية مرجّح (صداع انفجاري مفاجئ ± قيء/تدهور وعي)",
            "rationale": "صداع انفجاري مفاجئ مع قيء/تدهور وعي (وخاصة مع ارتفاع ضغط شديد) نمط يرفع احتمال النزف داخل القحف.",
            "safety": "حالة إسعافية: يلزم تصوير CT فوري وتدبير عاجل (خطر تدهور سريع)."
        }

    if has_focal and (ongoing or (dur_min is not None and dur_min >= 60) or _has_any(t, _ISCH_KEYWORDS)):
        return {
            "dx": "سكتة دماغية إقفارية حادة مرجّحة (أعراض عصبية بؤرية مفاجئة مستمرة ضمن نافذة زمنية قصيرة)",
            "rationale": "وجود أعراض عصبية بؤرية مفاجئة (ضعف/حبسة/عسر كلام…) مستمرة أو لمدة طويلة يدعم السكتة الإقفارية الحادة.",
            "safety": "حالة إسعافية: يلزم تقييم عاجل لأن نافذة العلاج (thrombolysis/thrombectomy) تعتمد على الوقت."
        }

    if has_focal and _has_any(t, _TIA_KEYWORDS):
        return {
            "dx": "نوبة إقفارية عابرة (TIA) محتملة",
            "rationale": "وجود أعراض عصبية بؤرية مع ذكر TIA/Transient ischemic attack يدعم احتمال النوبة العابرة.",
            "safety": "يلزم تقييم سريع لمنع سكتة لاحقة."
        }

    if _has_any(t, _HEM_KEYWORDS):
        return {
            "dx": "نزف داخل القحف/نزف دماغي محتمل",
            "rationale": "ذكر نزف/hemorrhage أو مفاهيم مرتبطة به يرفع احتمال النزف كسبب للأعراض.",
            "safety": "يلزم تدبير إسعافي وتصوير عاجل."
        }

    has_posterior = _has_any(t, _VERTEBROBASILAR_SYMPTOMS)
    if has_posterior and (ongoing or (dur_min is not None and dur_min >= 30)):
        return {
            "dx": "سكتة في الدورة الدموية الخلفية (جذع الدماغ/مخيخ) محتملة",
            "rationale": "وجود أعراض مثل الدوار، ازدواج الرؤية، وصعوبة المشي تدل على احتمال إصابة الدورة الخلفية.",
            "safety": "حالة إسعافية: قد تتدهور بسرعة، يلزم تصوير مقطعي وعائي أو رنين مغناطيسي."
        }

    return None


# =========================
# Grounding helpers
# =========================

_LLM_SKIP_WORDS = (
    "okay", "let me", "first", "so", "well", "the patient", "based on",
    "in this case", "the key", "the main", "the symptoms", "this patient",
    "we need to", "it is important to", "the most likely",
    "the differential diagnosis", "the provisional diagnosis",
    "the diagnosis is", "the context", "the context includes",
    "the provided context", "the case", "the case describes",
    "the patient presents", "the clinical picture", "the presentation",
    "the history", "the examination", "the findings", "the results",
    "the data", "the information", "the literature", "the evidence",
    "the study", "the report", "the article", "the abstract", "the paper",
    "the research", "the author", "the authors", "the group", "the team",
    "the institution"
)

_DIAGNOSTIC_KEYWORDS = (
    "diagnosis", "likely", "probable", "presents with", "suggests",
    "indicates", "consistent with", "compatible with",
    "is diagnosed", "is considered", "is suspected",
    "is most consistent with", "is likely", "is probable",
    "is the most likely", "is the probable", "is the diagnosis",
    "would be", "may be", "could be", "should be considered",
    "points to", "leads to", "results in", "causes",
    "manifests as", "presents as", "is characterized by",
    "is defined by", "is associated with", "is linked to",
    "is related to", "is due to", "is caused by",
    "is secondary to", "is the result of", "is the consequence of"
)

_UNCERTAINTY_TERMS = [
    "insufficient", "unclear", "cannot determine", "unable to determine",
    "not enough evidence", "limited evidence", "inconclusive",
    "unspecified", "unknown", "uncertain"
]


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _compute_source_grounding_strength(source: Dict) -> float:
    text = (source.get("text") or source.get("snippet") or "").lower()
    hybrid = _safe_float(source.get("hybrid_score"), 0.0)
    medcpt = source.get("medcpt_score")
    cosine = _safe_float(source.get("match_cosine"), 0.0)
    graph_score = _safe_float(source.get("graph_score"), 0.0)

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
        medcpt_val = _safe_float(medcpt, 0.0)
        if medcpt_val >= 0.80:
            strength += 0.25
        elif medcpt_val >= 0.60:
            strength += 0.18
        elif medcpt_val >= 0.40:
            strength += 0.10

    if cosine >= 0.75:
        strength += 0.25
    elif cosine >= 0.60:
        strength += 0.18
    elif cosine >= 0.45:
        strength += 0.10
    elif cosine >= 0.30:
        strength += 0.05

    if graph_score >= 0.70:
        strength += 0.16
    elif graph_score >= 0.45:
        strength += 0.10

    return round(strength, 4)


def _assess_grounding(case_text: str, sources: List[Dict]) -> Dict[str, object]:
    if not sources:
        return {
            "allow_llm": False,
            "reason": "no_sources",
            "top_strength": 0.0,
            "support_count": 0,
            "avg_strength_top3": 0.0,
        }

    strengths = []
    for s in sources:
        strength = _compute_source_grounding_strength(s)
        s["_grounding_strength"] = strength
        strengths.append(strength)

    strengths_sorted = sorted(strengths, reverse=True)
    top_strength = strengths_sorted[0] if strengths_sorted else 0.0
    avg_top3 = sum(strengths_sorted[:3]) / max(min(len(strengths_sorted), 3), 1)
    support_count = sum(1 for x in strengths_sorted if x >= 0.45)

    case_low = (case_text or "").lower()
    acute_or_high_risk = (
        _has_any(case_low, _HEM_PATTERN)
        or _has_any(case_low, _HEM_KEYWORDS)
        or _has_any(case_low, _BRAINSTEM_SYMPTOMS)
        or _has_any(case_low, _VERTEBROBASILAR_SYMPTOMS)
        or _has_any(case_low, _NEURO_FOCAL)
    )

    if acute_or_high_risk:
        allow = (top_strength >= 0.60) and (support_count >= 1 or avg_top3 >= 0.52)
        reason = "high_risk_grounded" if allow else "high_risk_weak_grounding"
    else:
        allow = (top_strength >= 0.52) and (support_count >= 1 or avg_top3 >= 0.45)
        reason = "grounded" if allow else "weak_grounding"

    return {
        "allow_llm": allow,
        "reason": reason,
        "top_strength": round(top_strength, 4),
        "support_count": int(support_count),
        "avg_strength_top3": round(avg_top3, 4),
    }


def _select_llm_context_sources(sources: List[Dict], limit: int = 4) -> List[Dict]:
    ranked = sorted(
        sources,
        key=lambda s: (
            _safe_float(s.get("_grounding_strength"), 0.0),
            _safe_float(s.get("graph_score"), 0.0),
            _safe_float(s.get("medcpt_score"), -1e9),
            _safe_float(s.get("match_cosine"), -1e9),
            _safe_float(s.get("hybrid_score"), -1e9),
        ),
        reverse=True,
    )

    selected = []
    seen_keys = set()

    for s in ranked:
        pmid = s.get("pmid")
        title = s.get("title", "")
        key = f"{pmid}::{title}"
        if key in seen_keys:
            continue
        seen_keys.add(key)

        txt = (s.get("text") or s.get("snippet") or "").strip()
        if not txt:
            continue

        selected.append(s)
        if len(selected) >= limit:
            break

    return selected


def _extract_llm_diagnosis(full_answer: str) -> str:
    lines = full_answer.split('.')
    best_line = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if any(line.lower().startswith(word) for word in _LLM_SKIP_WORDS):
            continue

        if any(kw in line.lower() for kw in _DIAGNOSTIC_KEYWORDS):
            return line.strip()

        if best_line is None:
            best_line = line.strip()

    if best_line is not None:
        return best_line

    return full_answer.strip()[:200]


def _llm_answer_is_safe(answer: str) -> bool:
    ans = (answer or "").strip().lower()
    if not ans:
        return False
    if any(t in ans for t in _UNCERTAINTY_TERMS):
        return False
    if len(ans) > 220:
        return False
    if ans.count("\n") > 2:
        return False
    return True


# =========================
# Answer formatting
# =========================

def _format_output(dx_pack: Dict[str, str], best_doc: Optional[Dict]) -> str:
    if not dx_pack or not best_doc:
        return "Evidence is insufficient"

    pmid = best_doc.get("pmid")
    title = best_doc.get("title") or "PubMed abstract"
    hs = best_doc.get("hybrid_score")
    mc = best_doc.get("match_cosine")
    ms = best_doc.get("medcpt_score")
    gs = best_doc.get("graph_score")
    gd = best_doc.get("graph_disease")

    lines = [
        f"- التشخيص: {dx_pack['dx']}",
        f"- المرجع الأقرب: [PMID:{pmid}] {title}",
        "- السكورز:",
        f"  - hybrid_score: {hs}",
        f"  - match_cosine: {mc}",
        f"  - medcpt_score: {ms}",
    ]

    if gs is not None:
        lines.append(f"  - graph_score: {gs}")
    if gd:
        lines.append(f"  - graph_disease: {gd}")

    return "\n".join(lines)


def _kg_priority(d: Dict) -> float:
    graph_score = _safe_float(d.get("graph_score"), 0.0)
    medcpt_score = _safe_float(d.get("medcpt_score"), -999.0)
    hybrid_score = _safe_float(d.get("score"), 0.0)

    kg_bonus = 0.0

    if graph_score >= 0.90:
        kg_bonus += 8.0
    elif graph_score >= 0.80:
        kg_bonus += 6.0
    elif graph_score >= 0.60:
        kg_bonus += 4.0
    elif graph_score >= 0.40:
        kg_bonus += 2.0

    if d.get("from_kg"):
        kg_bonus += 2.0

    return medcpt_score + kg_bonus + (0.35 * hybrid_score)


def _final_source_priority(s: Dict) -> float:
    graph_score = _safe_float(s.get("graph_score"), 0.0)
    medcpt = _safe_float(s.get("medcpt_score"), -999.0)
    hybrid = _safe_float(s.get("hybrid_score"), 0.0)
    cosine = _safe_float(s.get("match_cosine"), 0.0)

    kg_bonus = 0.0

    if graph_score >= 0.90:
        kg_bonus += 8.0
    elif graph_score >= 0.80:
        kg_bonus += 6.0
    elif graph_score >= 0.60:
        kg_bonus += 4.0
    elif graph_score >= 0.40:
        kg_bonus += 2.0

    if s.get("from_kg"):
        kg_bonus += 2.0

    return medcpt + kg_bonus + (0.35 * hybrid) + (0.25 * cosine)


def _dedup_docs(items: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    seen = set()

    for item in items:
        pmid = item.get("pmid")
        chunk_id = item.get("chunk_id")
        title = item.get("title", "")
        key = str(chunk_id) if chunk_id else f"{pmid}::{title}"

        if key in seen:
            continue

        seen.add(key)
        out.append(item)

    return out


# =========================
# Main function
# =========================

def get_answer(
    case_text: str,
    top_k: int | None = None,
    debug: bool = False,
    stroke_features: Optional[Dict] = None,
    use_rules: bool = True,
    use_llm: bool = False,
) -> Dict:
    case_text_en = case_text

    if debug:
        print(f"Case text: {case_text}")

    raw_k = top_k if top_k is not None else (CFG.get("retrieval") or {}).get("top_k_merged", 6)
    try:
        k = int(str(raw_k).replace("–", "-").split("-")[0].strip())
    except Exception:
        k = 6

    pm_cfg = CFG.get("pubmed", {}) or {}
    merged_docs: List[Dict] = []

    if pm_cfg.get("merge_with_local_kb", True) and _HAS_LOCAL:
        try:
            # مهم: نمرّر debug حتى نعرف من داخل pipeline هل KG رجع أم لا.
            local = hybrid_search(
                case_text_en,
                top_k=max(k * 3, 15),
                debug=debug
            )

            if debug:
                local_kg_count = sum(
                    1 for d in local
                    if d.get("source") == "GraphKB" or d.get("graph_score") is not None
                )
                print(f"PIPELINE local results count = {len(local)}")
                print(f"PIPELINE local KG results count = {local_kg_count}")
                if local:
                    print(
                        "PIPELINE local preview =",
                        [
                            {
                                "pmid": d.get("pmid"),
                                "title": d.get("title"),
                                "source": d.get("source"),
                                "graph_score": d.get("graph_score"),
                                "graph_disease": d.get("graph_disease"),
                            }
                            for d in local[:5]
                        ],
                    )

            for d in local:
                is_from_kg = d.get("source") == "GraphKB" or d.get("graph_score") is not None

                merged_docs.append({
                    "pmid": d.get("pmid"),
                    "doi": d.get("doi"),
                    "title": d.get("title", ""),
                    "text": d.get("text") or d.get("snippet", ""),
                    "url": d.get("source", ""),
                    "score": float(d.get("score", 0.0) or 0.0),
                    "snippet": safe_truncate(d.get("snippet") or d.get("text") or "", 300),
                    "chunk_id": d.get("chunk_id"),
                    "graph_score": d.get("graph_score"),
                    "graph_disease": d.get("graph_disease"),
                    "graph_matched_symptoms": d.get("graph_matched_symptoms", []),
                    "graph_category": d.get("graph_category"),
                    "graph_article_type": d.get("graph_article_type"),
                    "graph_acuity": d.get("graph_acuity"),
                    "graph_query_flags": d.get("graph_query_flags"),
                    "from_kg": is_from_kg,
                })

        except Exception as e:
            if debug:
                print("Local search error:", e)

    if not merged_docs:
        return {
            "answer": "Evidence is insufficient",
            "confidence": "منخفض",
            "top_score": 0.0,
            "sources": [],
            "pipeline_version": PIPELINE_VERSION,
            "used_fallback": True,
            "match_top": None,
        }

    merged_docs = _dedup_docs(merged_docs)

    if debug:
        merged_kg_count = sum(
            1 for d in merged_docs
            if d.get("from_kg") or d.get("graph_score") is not None
        )
        print(f"PIPELINE merged_docs count = {len(merged_docs)}")
        print(f"PIPELINE merged_docs KG count before rerank = {merged_kg_count}")

    merged_docs.sort(key=_kg_priority, reverse=True)

    try:
        use_medcpt = os.getenv("MEDRAG_USE_MEDCPT_RERANK", "1").strip() == "1"
        if use_medcpt:
            top_n = int(os.getenv("MEDRAG_MEDCPT_TOPN", "50"))

            # نحمي KG قبل rerank: نضمن أن KG يدخل ضمن head ولا يبقى بالـ tail.
            kg_docs_all = [
                d for d in merged_docs
                if d.get("from_kg") or d.get("graph_score") is not None
            ]
            non_kg_docs_all = [
                d for d in merged_docs
                if not (d.get("from_kg") or d.get("graph_score") is not None)
            ]

            kg_docs_all = sorted(kg_docs_all, key=_kg_priority, reverse=True)
            non_kg_docs_all = sorted(non_kg_docs_all, key=_kg_priority, reverse=True)

            protected_kg = kg_docs_all[:min(len(kg_docs_all), max(5, k))]
            rest_for_head = non_kg_docs_all[:max(0, top_n - len(protected_kg))]
            head = _dedup_docs(protected_kg + rest_for_head)

            used_keys = {
                str(d.get("chunk_id")) if d.get("chunk_id") else f"{d.get('pmid')}::{d.get('title', '')}"
                for d in head
            }

            tail = []
            for d in merged_docs:
                key = str(d.get("chunk_id")) if d.get("chunk_id") else f"{d.get('pmid')}::{d.get('title', '')}"
                if key not in used_keys:
                    tail.append(d)

            head = rerank_docs(case_text_en, head, text_key="text")

            # لا نسمح لـ MedCPT السلبي أن يقتل KG سريريًا.
            for d in head:
                if d.get("from_kg") or d.get("graph_score") is not None:
                    gs = _safe_float(d.get("graph_score"), 0.0)
                    floor_score = 5.0
                    if gs >= 0.90:
                        floor_score = 7.5
                    elif gs >= 0.80:
                        floor_score = 6.5
                    elif gs >= 0.60:
                        floor_score = 5.5

                    d["medcpt_score"] = max(
                        _safe_float(d.get("medcpt_score"), -999.0),
                        floor_score
                    )

            head = sorted(head, key=_kg_priority, reverse=True)
            merged_docs = _dedup_docs(head + tail)

    except Exception as e:
        if debug:
            print("MEDCPT_RERANK failed:", repr(e))

    if debug:
        merged_kg_count_after = sum(
            1 for d in merged_docs
            if d.get("from_kg") or d.get("graph_score") is not None
        )
        print(f"PIPELINE merged_docs KG count after rerank = {merged_kg_count_after}")
        print(
            "PIPELINE merged top preview =",
            [
                {
                    "pmid": d.get("pmid"),
                    "title": d.get("title"),
                    "score": d.get("score"),
                    "medcpt_score": d.get("medcpt_score"),
                    "graph_score": d.get("graph_score"),
                    "graph_disease": d.get("graph_disease"),
                    "from_kg": d.get("from_kg"),
                }
                for d in merged_docs[:10]
            ],
        )

    # نحضّر source_docs من كامل merged_docs وليس أول k*2 فقط؛
    # لأن KG قد يهبط بعد rerank إذا كانت نصوصه قصيرة أو MedCPT سلبي.
    kg_docs = [
        d for d in merged_docs
        if d.get("from_kg") or d.get("graph_score") is not None
    ]
    non_kg_docs = [
        d for d in merged_docs
        if not (d.get("from_kg") or d.get("graph_score") is not None)
    ]

    kg_docs = sorted(kg_docs, key=_kg_priority, reverse=True)
    non_kg_docs = sorted(non_kg_docs, key=_kg_priority, reverse=True)

    if kg_docs:
        source_doc_limit = max(k * 6, 30)
        forced_docs_count = min(len(kg_docs), max(5, k))
        source_docs = _dedup_docs(
            kg_docs[:forced_docs_count]
            + non_kg_docs[:source_doc_limit]
            + kg_docs[forced_docs_count:source_doc_limit]
        )
    else:
        source_docs = merged_docs[:max(k * 6, 30)]

    sources: List[Dict] = []

    for d in source_docs:
        src = {
            "title": d.get("title", ""),
            "source": d.get("url", ""),
            "pmid": d.get("pmid"),
            "doi": d.get("doi"),
            "hybrid_score": float(d.get("score", 0.0) or 0.0),
            "medcpt_score": d.get("medcpt_score"),
            "snippet": d.get("snippet") or safe_truncate(d.get("text", ""), 300),
            "text": d.get("text") or "",
            "chunk_id": d.get("chunk_id"),
            "graph_score": d.get("graph_score"),
            "graph_disease": d.get("graph_disease"),
            "graph_matched_symptoms": d.get("graph_matched_symptoms", []),
            "graph_category": d.get("graph_category"),
            "graph_article_type": d.get("graph_article_type"),
            "graph_acuity": d.get("graph_acuity"),
            "graph_query_flags": d.get("graph_query_flags"),
            "from_kg": bool(d.get("from_kg") or d.get("graph_score") is not None),
        }
        sources.append(src)

    if EMB is not None and sources:
        try:
            case_vec = EMB.encode([case_text_en], convert_to_numpy=True)[0]
            doc_texts = [s.get("text") or s.get("snippet") or "" for s in sources]
            doc_vecs = EMB.encode(doc_texts, convert_to_numpy=True)

            for s, v in zip(sources, doc_vecs):
                cos = _cosine_sim(case_vec, v)
                cos = max(-1.0, min(1.0, cos))
                s["match_cosine"] = float(round(cos, 4))
                s["match_percent"] = float(round((cos * 100.0) if cos >= 0 else 0.0, 2))

        except Exception:
            pass

    kg_sources = [
        s for s in sources
        if s.get("from_kg") or s.get("graph_score") is not None
    ]

    non_kg_sources = [
        s for s in sources
        if not (s.get("from_kg") or s.get("graph_score") is not None)
    ]

    kg_sources = sorted(kg_sources, key=_final_source_priority, reverse=True)
    non_kg_sources = sorted(non_kg_sources, key=_final_source_priority, reverse=True)

    if kg_sources:
        # المطلوب: فرض 1-2 KG على الأقل إذا موجودة.
        # هنا نجعلها 2 افتراضيًا، ولا نلغي FAISS/BM25.
        forced_kg_count = min(len(kg_sources), 2, k)
        mixed_sources = kg_sources[:forced_kg_count] + non_kg_sources

        remaining_kg = kg_sources[forced_kg_count:]
        mixed_sources = _dedup_docs(mixed_sources + remaining_kg)

        sources = sorted(mixed_sources, key=_final_source_priority, reverse=True)

        # حماية نهائية: إذا الترتيب أسقط KG من أول k، نحقنها يدويًا.
        final_sources = sources[:k]
        final_has_kg = any(
            s.get("from_kg") or s.get("graph_score") is not None
            for s in final_sources
        )

        if not final_has_kg and kg_sources:
            final_sources = kg_sources[:forced_kg_count] + final_sources
            final_sources = _dedup_docs(final_sources)[:k]

        sources = final_sources
    else:
        sources = non_kg_sources[:k]

    if debug:
        final_kg_count = sum(
            1 for s in sources
            if s.get("from_kg") or s.get("graph_score") is not None
        )
        print(f"PIPELINE final sources count = {len(sources)}")
        print(f"PIPELINE final KG sources count = {final_kg_count}")
        print(
            "PIPELINE final sources preview =",
            [
                {
                    "pmid": s.get("pmid"),
                    "title": s.get("title"),
                    "source": s.get("source"),
                    "hybrid_score": s.get("hybrid_score"),
                    "medcpt_score": s.get("medcpt_score"),
                    "graph_score": s.get("graph_score"),
                    "graph_disease": s.get("graph_disease"),
                    "from_kg": s.get("from_kg"),
                }
                for s in sources[:10]
            ],
        )

    grounding = _assess_grounding(case_text_en, sources)

    match_top = None
    if sources:
        kg_candidates = [
            s for s in sources
            if (s.get("from_kg") or s.get("graph_score") is not None)
            and _safe_float(s.get("graph_score"), 0.0) >= 0.60
        ]

        if kg_candidates:
            best = max(kg_candidates, key=_final_source_priority)
        else:
            best = max(sources, key=_final_source_priority)

        match_top = {
            "pmid": best.get("pmid"),
            "title": best.get("title"),
            "medcpt_score": best.get("medcpt_score"),
            "match_cosine": best.get("match_cosine"),
            "match_percent": best.get("match_percent"),
            "hybrid_score": best.get("hybrid_score"),
            "graph_score": best.get("graph_score"),
            "graph_disease": best.get("graph_disease"),
            "graph_category": best.get("graph_category"),
            "graph_article_type": best.get("graph_article_type"),
            "graph_acuity": best.get("graph_acuity"),
            "from_kg": best.get("from_kg"),
            "grounding_strength": best.get("_grounding_strength"),
        }

    best_doc = match_top if match_top else (sources[0] if sources else None)

    dx_pack = None
    if use_rules:
        dx_pack = _rule_diagnose_from_case_v2(case_text_en)
        if dx_pack is None:
            dx_pack = _rule_diagnose_from_case(case_text_en)

    answer = "Evidence is insufficient"
    used_fallback = True

    if dx_pack and best_doc:
        answer = _format_output(dx_pack, best_doc)
        used_fallback = False

    elif use_llm and sources:
        if grounding.get("allow_llm", False):
            try:
                llm_sources = _select_llm_context_sources(sources, limit=4)
                if llm_sources:
                    context_blocks = []
                    for i, s in enumerate(llm_sources, start=1):
                        context_blocks.append(
                            f"[{i}] PMID:{s.get('pmid')} | Title: {s.get('title')}\n{s.get('text')}"
                        )

                    context = "\n\n".join(context_blocks)
                    user_prompt = USER_PROMPT_TEMPLATE.format(case=case_text_en, context=context)

                    llm_provider = os.getenv(
                        "LLM_PROVIDER",
                        os.getenv("llm_provider", (CFG.get("models") or {}).get("llm_provider", "local")),
                    ).lower()

                    llm_model = (CFG.get("models") or {}).get("llm_model")

                    full_answer = generate_answer(
                        SYSTEM_PROMPT,
                        user_prompt,
                        provider=llm_provider,
                        model_name=llm_model,
                    )

                    extracted = _extract_llm_diagnosis(full_answer)
                    if _llm_answer_is_safe(extracted):
                        answer = extracted
                    else:
                        answer = "Evidence is insufficient"

                    used_fallback = True

                else:
                    answer = "Evidence is insufficient"
                    used_fallback = True

            except Exception as e:
                if debug:
                    print("LLM generation failed:", repr(e))
                answer = "Evidence is insufficient"
                used_fallback = True
        else:
            answer = "Evidence is insufficient"
            used_fallback = True
    else:
        answer = "Evidence is insufficient"
        used_fallback = True

    top_score = float(merged_docs[0].get("score", 0.0) or 0.0) if merged_docs else 0.0
    confidence = "متوسط" if top_score >= 0.35 else "منخفض"

    if answer == "Evidence is insufficient":
        confidence = "منخفض"
    elif grounding.get("top_strength", 0.0) >= 0.72:
        confidence = "مرتفع"
    elif grounding.get("top_strength", 0.0) >= 0.55:
        confidence = "متوسط"

    stroke_risk = None
    if stroke_features is not None and _HAS_STROKE_MODEL and predict_stroke_risk is not None:
        try:
            stroke_risk = predict_stroke_risk(stroke_features)
        except Exception:
            stroke_risk = None

    if debug:
        try:
            pm = best_doc.get("pmid") if best_doc else None
            ti = best_doc.get("title") if best_doc else None
            print(f"PIPELINE_VERSION = {PIPELINE_VERSION}")
            print(f"TOP_DOC = {pm} {ti}")
            print("ANSWER_PREVIEW:\n", answer[:900])
            print("match_top =", match_top)
            print("grounding =", grounding)
        except Exception:
            pass

    out = {
        "answer": answer,
        "confidence": confidence,
        "top_score": top_score,
        "sources": sources,
        "pipeline_version": PIPELINE_VERSION,
        "used_fallback": used_fallback,
        "match_top": match_top,
        "grounding": grounding,
    }

    if stroke_risk is not None:
        out["stroke_risk"] = stroke_risk

    return out


def get_stroke_answer(
    case_text: str,
    stroke_features: Dict,
    top_k: int | None = None,
    debug: bool = False,
) -> Dict:
    if not _HAS_STROKE_MODEL or predict_stroke_risk is None:
        raise RuntimeError(
            "stroke_risk model غير متاح. تأكد من وجود app/models/stroke_risk.py وملف الموديل joblib."
        )

    return get_answer(
        case_text,
        top_k=top_k,
        debug=debug,
        stroke_features=stroke_features,
    )
