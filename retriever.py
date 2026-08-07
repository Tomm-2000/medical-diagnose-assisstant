# -*- coding: utf-8 -*-
# app/rag/retriever.py

from __future__ import annotations

from pathlib import Path
import json
import re
import time
import yaml
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import faiss
from sentence_transformers import SentenceTransformer
import joblib

from app.rag.utils_text import safe_truncate
from app.rag.clinical_signals import detect_clinical_signals, build_dynamic_query_expansions, summarize_signals
from app.rag.hybrid_feedback import build_hybrid_feedback, search_index_with_vector
from app.rag.common_utils import _contains_arabic

# ---------------- Config ----------------
with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f) or {}

EMBED_MODEL = (CFG.get("models") or {}).get(
    "embeddings",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

try:
    import torch  # type: ignore
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    _DEVICE = "cpu"

EMB = SentenceTransformer(EMBED_MODEL, device=_DEVICE)

IDX_PATH = Path("app/index/faiss_hnsw.index")
METAS_PATH = Path("app/index/metas.json")
BM25_PATH = Path("app/index/bm25.joblib")

if not IDX_PATH.exists():
    raise FileNotFoundError(f"FAISS index not found: {IDX_PATH}")
if not METAS_PATH.exists():
    raise FileNotFoundError(f"metas.json not found: {METAS_PATH}")

_index = faiss.read_index(str(IDX_PATH))
_metas = json.loads(METAS_PATH.read_text(encoding="utf-8"))

USE_BM25 = BM25_PATH.exists()

_bm25 = None
if USE_BM25 and BM25_PATH.exists():
    pack = joblib.load(BM25_PATH)
    _bm25 = pack.get("bm25")

# ========== Knowledge Graph Integration ==========
KG_CFG = CFG.get("knowledge_graph", {})
USE_KG = bool(KG_CFG.get("enabled", False))

KG_URI_IN_USE = None
_KG_INIT_ATTEMPTED = False
_KG_LAST_ERROR = None

# ========== Hybrid Feedback / Pseudo-Relevance Feedback ==========
# This is the v7.10 hybrid retrieval upgrade. It is deliberately optional and
# has a full fallback to the existing v7.8/v7.9 retrieval path if anything fails.
HYB_FB_CFG = (CFG.get("hybrid_feedback") or CFG.get("semantic_feedback") or {})
USE_HYBRID_FEEDBACK = bool(HYB_FB_CFG.get("enabled", False))

# ---------------- Regex / Tokens ----------------
_AR_RE = re.compile(r"[\u0600-\u06FF]")
_PMID_RE = re.compile(r"\bPMID:\s*(\d{6,9})\b", re.IGNORECASE)
_DOI_RE = re.compile(r"\bdoi:\s*([0-9.]+/[^\s]+)", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z0-9]+|[\u0600-\u06FF]+", re.UNICODE)

# ---------------- Terms ----------------
ACUTE_HINT_TERMS = [
    "acute", "sudden", "suddenly", "onset", "new onset", "acute onset",
    "abrupt", "abruptly", "within", "hour", "hours", "minute", "minutes",
    "min", "mins", "hr", "hrs", "today", "this morning", "this afternoon",
    "this evening", "last night", "same day",
    "nihss",
    "aphasia", "dysarthria", "speech", "slurred",
    "weakness", "unilateral weakness", "arm weakness", "leg weakness",
    "hemiparesis", "hemiplegia", "facial droop",
    "tia", "transient ischemic attack",
    "thrombolysis", "alteplase", "tpa",
    "thrombectomy", "endovascular",
    "large vessel", "lvo", "occlusion",
    "ct", "cta", "mri", "diffusion", "dwi",
    "emergency", "prehospital",
    "fast", "act fast",
]

MOTOR_TERMS = [
    "weakness", "unilateral weakness", "arm weakness", "leg weakness",
    "hemiparesis", "hemiplegia",
]

LANGUAGE_TERMS = [
    "aphasia", "speech disturbance", "speech", "dysarthria", "slurred",
]

POSTERIOR_TERMS = [
    "vertigo", "diplopia", "double vision", "ataxia", "dizziness",
    "gait difficulty", "difficulty walking", "nystagmus", "imbalance",
    "brainstem", "cerebellar", "posterior circulation",
]

HEMORRHAGE_TERMS = [
    "hemorrhage", "haemorrhage", "bleeding", "intracerebral hemorrhage",
    "subarachnoid hemorrhage", "sah", "ich", "thunderclap", "worst headache",
    "vomiting", "loss of consciousness", "decreased consciousness",
    "worsening consciousness",
]

COMPLICATION_TOPICS = [
    "delirium",
    "depression",
    "poststroke depression",
    "post-stroke depression",
    "poststroke delirium",
    "post-stroke delirium",
    "seizure",
    "epilepsy",
    "poststroke seizure",
    "post-stroke seizure",
    "poststroke epilepsy",
    "post-stroke epilepsy",
]

CHRONIC_OR_GENERAL_PENALTY_TERMS = [
    "cognitive impairment", "vascular cognitive impairment", "dementia", "alzheimer",
    "long-term", "long term", "survivor", "survivors", "rehabilitation",
    "risk factor", "risk factors", "genetic", "genome", "locus", "loci",
    "registry", "burden", "incidence", "prevalence", "epidemiology",
    "systematic review", "meta-analysis", "meta analysis",
    "chronic", "gradual", "gradually", "progressive", "progressively",
    "for weeks", "for months", "for years", "over weeks", "over months",
    "over years", "several weeks", "several months", "several years",
]

DEFAULT_STROKE_EXPANSION = (
    "acute ischemic stroke sudden onset hemiparesis aphasia dysarthria "
    "unilateral weakness facial droop speech disturbance within 1 hour TIA NIHSS "
    "thrombolysis alteplase tPA thrombectomy endovascular large vessel occlusion CT CTA MRI DWI "
    "emergency FAST act fast"
)

AR_ACUTE_HINTS = [
    "مفاجئ", "فجأة", "حاد", "خلال", "ساعة", "ساعات", "دقيقة", "دقائق",
    "ضعف", "شلل", "نصفي", "حبسة", "كلام", "نطق", "تلعثم",
    "تنميل", "دوخة", "إغماء", "وعي", "وجه", "ميلان",
]

# ---------------- Precomputed maps ----------------
_pmid_to_meta_indices: Dict[str, List[int]] = defaultdict(list)
for idx, meta in enumerate(_metas):
    pmid = meta.get("pmid")
    if pmid is not None:
        _pmid_to_meta_indices[str(pmid)].append(idx)


# ---------------- KG helpers ----------------
def _normalize_neo4j_uri(uri: str) -> str:
    u = (uri or "").strip()
    if not u:
        return "bolt://localhost:7687"

    if u.startswith("neo4j://"):
        return "bolt://" + u[len("neo4j://"):]
    if u.startswith("neo4j+s://"):
        return "bolt://" + u[len("neo4j+s://"):]
    if u.startswith("neo4j+ssc://"):
        return "bolt://" + u[len("neo4j+ssc://"):]

    return u


def _candidate_kg_uris(config_uri: str) -> List[str]:
    base = _normalize_neo4j_uri(config_uri)
    candidates = [base]

    if "localhost" in base:
        candidates.append(base.replace("localhost", "127.0.0.1"))

    if "127.0.0.1" in base:
        candidates.append(base.replace("127.0.0.1", "localhost"))

    if "7687" not in base:
        candidates.extend([
            "bolt://localhost:7687",
            "bolt://127.0.0.1:7687",
        ])
    else:
        if "bolt://localhost:7687" not in candidates:
            candidates.append("bolt://localhost:7687")
        if "bolt://127.0.0.1:7687" not in candidates:
            candidates.append("bolt://127.0.0.1:7687")

    return list(dict.fromkeys(candidates))


def _instantiate_graph_retriever(uri: str):
    from app.knowledge_graph.graph_retriever import GraphRetriever

    clean_uri = _normalize_neo4j_uri(str(uri or "bolt://localhost:7687")).strip()
    user = str(KG_CFG.get("user", "neo4j") or "neo4j").strip()
    password = str(KG_CFG.get("password", "password") or "password").strip()

    if clean_uri in {"neo4j", "password", user, password}:
        raise ValueError(
            f"Invalid KG URI received by _instantiate_graph_retriever: {clean_uri!r}"
        )

    if not (
        clean_uri.startswith("bolt://")
        or clean_uri.startswith("bolt+s://")
        or clean_uri.startswith("bolt+ssc://")
    ):
        raise ValueError(
            f"Invalid KG URI after normalization: {clean_uri!r}"
        )

    return GraphRetriever(
        uri=clean_uri,
        user=user,
        password=password,
    )


def _search_kg_fresh(query: str, top_k: int, debug: bool = False) -> List[Dict]:
    """
    ينشئ اتصالًا جديدًا لكل عملية KG search لتجنب مشاكل session/socket lifecycle.
    """
    global KG_URI_IN_USE, _KG_INIT_ATTEMPTED, _KG_LAST_ERROR

    if not USE_KG:
        return []

    _KG_INIT_ATTEMPTED = True
    candidate_uris = _candidate_kg_uris(KG_CFG.get("uri", "bolt://localhost:7687"))

    last_error = None

    for candidate_uri in candidate_uris:
        clean_uri = _normalize_neo4j_uri(str(candidate_uri or "bolt://localhost:7687")).strip()

        for attempt in range(2):
            try:
                if debug:
                    print(
                        "KG instantiate args =",
                        {
                            "uri": clean_uri,
                            "user": str(KG_CFG.get("user", "neo4j")),
                            "password_type": type(KG_CFG.get("password", "password")).__name__,
                        }
                    )

                gr = _instantiate_graph_retriever(clean_uri)
                KG_URI_IN_USE = clean_uri
                _KG_LAST_ERROR = None

                if debug:
                    print(f"✅ KG query via {clean_uri} (attempt {attempt + 1})")

                return gr.search(query, top_k=top_k)

            except Exception as e:
                last_error = e
                _KG_LAST_ERROR = e

                if debug:
                    print(f"⚠️ KG failed via {clean_uri} (attempt {attempt + 1}): {e!r}")

                time.sleep(0.25)

    if debug and last_error is not None:
        print(f"⚠️ فشل KG search نهائيًا: {last_error}")

    return []


# ---------------- Helpers ----------------
def _extract_time_context(query: str) -> Dict:
    q = (query or "").lower()

    ctx = {
        "has_time": False,
        "duration_minutes": None,
        "is_acute_time": False,
        "is_chronic_time": False,
    }

    chronic_terms = [
        "chronic", "gradual", "gradually", "progressive", "progressively",
        "long-standing", "longstanding",
        "for weeks", "for months", "for years",
        "over weeks", "over months", "over years",
        "several weeks", "several months", "several years",
    ]

    if any(t in q for t in chronic_terms):
        ctx["has_time"] = True
        ctx["is_chronic_time"] = True
        return ctx

    patterns = [
        r"\b(?:for|since|within|over|during|last|past)?\s*(\d+(?:\.\d+)?)\s*(minute|minutes|min|mins)\b",
        r"\b(?:for|since|within|over|during|last|past)?\s*(\d+(?:\.\d+)?)\s*(hour|hours|hr|hrs)\b",
        r"\b(?:for|since|within|over|during|last|past)?\s*(\d+(?:\.\d+)?)\s*(day|days)\b",
        r"\b(?:for|since|within|over|during|last|past)?\s*(\d+(?:\.\d+)?)\s*(week|weeks|month|months|year|years)\b",
    ]

    for p in patterns:
        m = re.search(p, q)
        if not m:
            continue

        value = float(m.group(1))
        unit = m.group(2)

        ctx["has_time"] = True

        if unit in {"minute", "minutes", "min", "mins"}:
            ctx["duration_minutes"] = value
            ctx["is_acute_time"] = True
        elif unit in {"hour", "hours", "hr", "hrs"}:
            ctx["duration_minutes"] = value * 60
            ctx["is_acute_time"] = True
        elif unit in {"day", "days"}:
            ctx["duration_minutes"] = value * 24 * 60
            ctx["is_acute_time"] = value <= 1
            ctx["is_chronic_time"] = value >= 14
        else:
            ctx["is_chronic_time"] = True

        return ctx

    vague_acute = [
        "sudden onset", "acute onset", "new onset", "abrupt onset",
        "started today", "began today", "developed today",
        "this morning", "this afternoon", "this evening",
        "last night", "today", "same day",
        "few minutes", "several minutes", "few hours", "several hours",
        "just started", "suddenly developed", "suddenly started",
    ]

    if any(t in q for t in vague_acute):
        ctx["has_time"] = True
        ctx["is_acute_time"] = True

    return ctx


def _query_is_acute(q: str) -> bool:
    """
    Backward-compatible acute detector.
    The legacy keyword logic is now wrapped by the clinical_signals layer,
    so the old terms remain a fallback/explainability source rather than
    the only decision layer.
    """
    t = (q or "").strip()
    if not t:
        return False

    try:
        return bool(detect_clinical_signals(t, use_model=False).get("is_acute"))
    except Exception:
        low = t.lower()
        time_ctx = _extract_time_context(low)

        if time_ctx.get("is_chronic_time"):
            return False
        if time_ctx.get("is_acute_time"):
            return True
        if _contains_arabic(t):
            return any(w in low for w in AR_ACUTE_HINTS)
        if any(w in low for w in ACUTE_HINT_TERMS):
            return True
        return any(w in low for w in ["stroke", "tia", "fast", "act"])


def _extract_ids(text: str):
    if not text:
        return None, None
    pm = _PMID_RE.search(text)
    doi = _DOI_RE.search(text)
    return (pm.group(1) if pm else None, doi.group(1) if doi else None)


def _to_similarity(dist_or_score: float) -> float:
    metric = getattr(_index, "metric_type", None)
    if metric == faiss.METRIC_L2:
        return float(1.0 - (dist_or_score / 2.0))
    return float(dist_or_score)


def _count_hits(text: str, terms: List[str]) -> int:
    t = (text or "").lower()
    return sum(1 for k in terms if k in t)


def _acute_bonus(text: str) -> float:
    hits = min(_count_hits(text, ACUTE_HINT_TERMS), 10)
    return 0.03 * hits


def _chronic_penalty(text: str) -> float:
    hits = min(_count_hits(text, CHRONIC_OR_GENERAL_PENALTY_TERMS), 10)
    return 0.03 * hits

def _extract_query_age(query_low: str) -> Optional[int]:
    patterns = [
        r"\b(\d{1,3})\s*[- ]?year[- ]old\b",
        r"\b(\d{1,3})\s*years?\s*old\b",
        r"\b(\d{1,3})\s*y/o\b",
        r"\b(\d{1,3})\s*yo\b",
    ]

    for pattern in patterns:
        m = re.search(pattern, query_low or "")
        if not m:
            continue

        try:
            age = int(m.group(1))
            if 0 <= age <= 120:
                return age
        except Exception:
            continue

    return None


def _query_is_adult(query_low: str) -> bool:
    age = _extract_query_age(query_low)

    return bool(
        (age is not None and age >= 18)
        or any(t in (query_low or "") for t in [
            "adult", "man", "woman", "male", "female",
            "elderly", "older adult", "old man", "old woman"
        ])
    )


def _article_has_pediatric_signal(title: str, text: str) -> bool:
    combined = f"{title or ''} {text or ''}".lower()

    return bool(
        any(t in combined for t in [
            "child",
            "children",
            "childhood",
            "pediatric",
            "paediatric",
            "infant",
            "newborn",
            "neonate",
            "adolescent",
            "boy",
            "girl",
        ])
        or re.search(r"\b([0-9]|1[0-7])\s*[- ]?year[- ]old\b", combined)
        or re.search(r"\b([0-9]|1[0-7])\s*years?\s*old\b", combined)
    )


def _article_has_fever_or_infection_signal(title: str, text: str) -> bool:
    combined = f"{title or ''} {text or ''}".lower()

    return any(t in combined for t in [
        "fever",
        "febrile",
        "infection",
        "infectious",
        "meningitis",
        "encephalitis",
        "cryptococcosis",
        "sepsis",
        "fungal",
        "bacterial",
        "viral",
    ])


def _query_has_fever_or_infection(query_low: str) -> bool:
    return any(t in (query_low or "") for t in [
        "fever",
        "febrile",
        "infection",
        "infectious",
        "meningitis",
        "encephalitis",
        "sepsis",
        "fungal",
        "bacterial",
        "viral",
    ])


def _should_skip_clinical_mismatch(title: str, text: str, query_low: str) -> bool:
    """
    Hard clinical gate:
    adult case should not use pediatric case reports as top evidence.
    """
    if _query_is_adult(query_low) and _article_has_pediatric_signal(title, text):
        return True

    return False


def _clinical_mismatch_penalty(title: str, text: str, query_low: str) -> float:
    """
    Soft penalty for clinically mismatched articles.
    This affects ranking but does not necessarily skip the article.
    """
    penalty = 0.0

    if _query_is_adult(query_low) and _article_has_pediatric_signal(title, text):
        penalty += 1.40

    if _article_has_fever_or_infection_signal(title, text) and not _query_has_fever_or_infection(query_low):
        penalty += 0.55

    return penalty


def _is_kg_item(item: Dict) -> bool:
    return bool(
        item.get("_from_graph")
        or item.get("graph_score") is not None
        or item.get("source") == "GraphKB"
    )


def _kg_raw_top_k(k: int) -> int:
    """
    Prevent KG from returning huge raw pools like 270 when pipeline asks for 90.
    """
    configured = (KG_CFG or {}).get("raw_top_k")
    if configured is not None:
        try:
            return max(12, int(configured))
        except Exception:
            pass

    return max(12, min(60, int(k)))


def _kg_return_cap(k: int) -> int:
    """
    Maximum KG items allowed in hybrid_search final output.
    For k=90 => about 30 KG max.
    For k=15 => about 5 KG max.
    """
    ratio = float((KG_CFG or {}).get("max_return_ratio", 0.35))
    hard_cap = int((KG_CFG or {}).get("max_return_cap", 30))

    cap = int(round(max(1, k) * ratio))
    cap = max(2, cap)
    cap = min(hard_cap, cap)
    cap = min(max(1, k), cap)

    return cap

def _faiss_search(query: str, k_candidates: int) -> List[Tuple[int, float]]:
    q_vec = EMB.encode([query], normalize_embeddings=True).astype("float32")
    D, I = _index.search(q_vec, k_candidates)
    return [(int(idx), float(val)) for idx, val in zip(I[0], D[0])]


def _bm25_tokenize(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def _bm25_search(query: str, k_candidates: int) -> List[Tuple[int, float]]:
    if _bm25 is None:
        return []
    q_tokens = _bm25_tokenize(query)
    if not q_tokens:
        return []
    scores = _bm25.get_scores(q_tokens)
    pairs = [(i, float(scores[i])) for i in range(len(scores))]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:k_candidates]


def _minmax_norm(pairs: List[Tuple[int, float]]) -> Dict[int, float]:
    if not pairs:
        return {}
    vals = [s for _, s in pairs]
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-9:
        return {i: 1.0 for i, _ in pairs}
    return {i: (s - mn) / (mx - mn) for i, s in pairs}


def _minmax_norm_scores(values: List[float]) -> List[float]:
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx - mn < 1e-9:
        return [1.0 for _ in values]
    return [(v - mn) / (mx - mn) for v in values]


def _best_meta_for_pmid(pmid: str, query_low: str) -> Optional[Dict]:
    indices = _pmid_to_meta_indices.get(str(pmid), [])
    if not indices:
        return None

    best_meta = None
    best_score = -1e9

    for idx in indices:
        meta = _metas[idx]
        text = meta.get("text", "") or ""
        text_low = text.lower()

        local_score = 0.0
        local_score += 0.5 * min(_count_hits(text, ACUTE_HINT_TERMS), 8)
        local_score += 0.6 * min(_count_hits(text, POSTERIOR_TERMS), 6)
        local_score += 0.6 * min(_count_hits(text, HEMORRHAGE_TERMS), 6)

        if "weakness" in query_low and any(t in text_low for t in MOTOR_TERMS):
            local_score += 2.0
        if any(t in query_low for t in ["speech", "dysarthria", "aphasia", "slurred"]):
            if any(t in text_low for t in LANGUAGE_TERMS):
                local_score += 2.0
        if any(t in query_low for t in POSTERIOR_TERMS):
            if any(t in text_low for t in POSTERIOR_TERMS):
                local_score += 2.5
        if any(t in query_low for t in HEMORRHAGE_TERMS):
            if any(t in text_low for t in HEMORRHAGE_TERMS):
                local_score += 2.5

        local_score += min(len(text) / 1500.0, 1.0)

        if local_score > best_score:
            best_score = local_score
            best_meta = meta

    return best_meta


# ---------------- Main API ----------------
def hybrid_search(query: str, top_k: int | None = None, debug: bool = False):
    q = (query or "").strip()
    if not q:
        return []

    k = int(top_k or int((CFG.get("retrieval") or {}).get("top_k_merged", 5)))

    retrieval_cfg = (CFG.get("retrieval") or {})
    w_dense = float(retrieval_cfg.get("w_dense", 0.6))
    w_bm25 = float(retrieval_cfg.get("w_bm25", 0.4))
    w_graph = float((KG_CFG or {}).get("graph_weight", 0.3))

    k_candidates_dense = 500
    k_candidates_bm25 = 1000
    keep_dense = 250
    keep_bm25 = 400

    q_low = q.lower()

    try:
        clinical_signals = detect_clinical_signals(q)
    except Exception:
        clinical_signals = detect_clinical_signals(q, use_model=False)

    query_is_acute = bool(clinical_signals.get("is_acute"))
    query_has_motor = bool(clinical_signals.get("has_motor"))
    query_has_language = bool(clinical_signals.get("has_language"))
    query_has_posterior = bool(clinical_signals.get("is_posterior"))
    query_has_hemorrhage = bool(clinical_signals.get("is_hemorrhage"))

    # v7.10: Hybrid pseudo-relevance feedback.
    # First-stage retrieval is collected from ALL available channels:
    #   - FAISS dense search
    #   - BM25 lexical search
    #   - Knowledge Graph search
    # Then the feedback layer builds:
    #   - a refined dense vector for second-stage FAISS,
    #   - an evidence-derived text query for second-stage BM25/KG.
    # Rule-based expansions remain fallback/safety only.
    queries = [q]
    feedback_query = ""
    hybrid_feedback_info = {
        "enabled": bool(USE_HYBRID_FEEDBACK),
        "used": False,
        "reason": "disabled",
    }

    use_rule_expansions = bool(
        (CFG.get("clinical_signals") or {}).get("use_dynamic_query_expansion", True)
    )
    rule_expansion_mode = str(
        HYB_FB_CFG.get("rule_expansion_mode", "fallback")
    ).strip().lower()

    dense_merged: Dict[int, float] = {}

    # Stage 1A — FAISS original-query seed.
    initial_dense_pairs = _faiss_search(q, k_candidates_dense)
    for idx, dist_or_score in initial_dense_pairs:
        if idx < 0 or idx >= len(_metas):
            continue
        sim = _to_similarity(dist_or_score)
        dense_merged[idx] = max(dense_merged.get(idx, -1e9), sim)

    # Stage 1B — BM25 original-query seed.
    initial_bm25_pairs: List[Tuple[int, float]] = []
    if USE_BM25 and _bm25 is not None:
        initial_bm25_pairs = _bm25_search(q, k_candidates_bm25)

    # Stage 1C — KG original-query seed.
    # We call it early only when hybrid feedback wants KG evidence, then reuse
    # the same graph_results later so Neo4j is not queried twice unnecessarily.
    kg_raw_limit = _kg_raw_top_k(k)
    graph_results: List[Dict] = []
    if USE_HYBRID_FEEDBACK and bool(HYB_FB_CFG.get("use_kg_feedback", True)):
        graph_results = _search_kg_fresh(q, top_k=kg_raw_limit, debug=debug)

    # Stage 2 — Build feedback from FAISS + BM25 + KG evidence.
    if USE_HYBRID_FEEDBACK:
        refined_vec, feedback_query, hybrid_feedback_info = build_hybrid_feedback(
            query=q,
            embedder=EMB,
            metas=_metas,
            dense_pairs=initial_dense_pairs[: int(HYB_FB_CFG.get("dense_seed_pool", 80))],
            bm25_pairs=initial_bm25_pairs[: int(HYB_FB_CFG.get("bm25_seed_pool", 80))],
            kg_results=graph_results[: int(HYB_FB_CFG.get("kg_seed_pool", 40))],
            pmid_to_meta_indices=_pmid_to_meta_indices,
            feedback_docs_per_channel=int(HYB_FB_CFG.get("feedback_docs_per_channel", 6)),
            alpha=float(HYB_FB_CFG.get("alpha", 0.75)),
            beta=float(HYB_FB_CFG.get("beta", 0.25)),
            max_doc_chars=int(HYB_FB_CFG.get("max_doc_chars", 1200)),
            max_terms=int(HYB_FB_CFG.get("max_terms", 14)),
            debug=debug,
        )

        if refined_vec is not None:
            feedback_pairs = search_index_with_vector(
                _index,
                refined_vec,
                int(HYB_FB_CFG.get("k_final_dense", k_candidates_dense)),
            )
            for idx, dist_or_score in feedback_pairs:
                if idx < 0 or idx >= len(_metas):
                    continue
                sim = _to_similarity(dist_or_score)
                dense_merged[idx] = max(
                    dense_merged.get(idx, -1e9),
                    sim + float(HYB_FB_CFG.get("feedback_dense_bonus", 0.03)),
                )

        # BM25/KG second-stage query is evidence-derived, not expert-template.
        if hybrid_feedback_info.get("used") and feedback_query and feedback_query != q:
            queries.append(feedback_query)

    hybrid_feedback_used = bool(hybrid_feedback_info.get("used"))

    should_apply_rule_expansions = (
        use_rule_expansions
        and (
            not USE_HYBRID_FEEDBACK
            or rule_expansion_mode == "always"
            or (rule_expansion_mode == "fallback" and not hybrid_feedback_used)
        )
    )

    if should_apply_rule_expansions:
        # v7.8 rule expansions are fallback/explainability. They are not the
        # primary retrieval mechanism when hybrid feedback succeeds.
        for expansion in build_dynamic_query_expansions(q, clinical_signals):
            queries.append(q + " " + expansion)

        if query_has_motor:
            queries += [
                q + " arm weakness",
                q + " leg weakness",
                q + " unilateral weakness",
                q + " hemiparesis",
            ]
        if query_has_language:
            queries += [
                q + " aphasia",
                q + " speech disturbance",
                q + " dysarthria",
                q + " slurred speech",
            ]
        if query_has_posterior:
            queries += [
                q + " posterior circulation stroke",
                q + " brainstem stroke",
                q + " cerebellar stroke",
                q + " vertigo diplopia ataxia",
            ]
        if query_has_hemorrhage:
            queries += [
                q + " intracerebral hemorrhage",
                q + " subarachnoid hemorrhage",
                q + " thunderclap headache vomiting decreased consciousness",
            ]

        if query_has_motor or query_has_language:
            queries += [q + " FAST", q + " act fast", q + " stroke symptoms", q + " facial droop"]

    queries = list(dict.fromkeys(queries))

    if debug:
        print("Clinical signals =", summarize_signals(clinical_signals))
        print("Hybrid feedback =", hybrid_feedback_info)
        print("Hybrid feedback query =", feedback_query[:500] if feedback_query else "")
        print("Rule expansion mode =", rule_expansion_mode)
        print("Rule expansions applied =", should_apply_rule_expansions)
        print("Query count for BM25/fallback dense =", len(queries))

    # If rule expansions are active, add their dense results too. If hybrid
    # feedback succeeded and mode=fallback, this loop only adds evidence-derived
    # feedback query results, not manual expert-system templates.
    for qq in queries:
        if qq == q:
            continue
        for idx, dist_or_score in _faiss_search(qq, k_candidates_dense):
            if idx < 0 or idx >= len(_metas):
                continue
            sim = _to_similarity(dist_or_score)
            dense_merged[idx] = max(dense_merged.get(idx, -1e9), sim)

    dense_pairs = sorted(dense_merged.items(), key=lambda x: x[1], reverse=True)[:keep_dense]
    dense_norm = _minmax_norm(dense_pairs)

    bm25_norm: Dict[int, float] = {}
    if USE_BM25 and _bm25 is not None:
        bm25_merged: Dict[int, float] = {}
        for qq in queries:
            for idx, sc in _bm25_search(qq, k_candidates_bm25):
                if idx < 0 or idx >= len(_metas):
                    continue
                bm25_merged[idx] = max(bm25_merged.get(idx, -1e9), sc)

        bm25_pairs = sorted(bm25_merged.items(), key=lambda x: x[1], reverse=True)[:keep_bm25]
        bm25_norm = _minmax_norm(bm25_pairs)

    all_ids = set(dense_norm.keys()) | set(bm25_norm.keys())

    results = []
    for idx in all_ids:
        meta = _metas[int(idx)]
        txt = meta.get("text", "") or ""
        txt_low = txt.lower()
        title = meta.get("title", "") or ""
        title_low = title.lower()

        pmid = meta.get("pmid")
        doi = meta.get("doi")
        if not pmid or not doi:
            pm2, d2 = _extract_ids(txt)
            pmid = pmid or pm2
            doi = doi or d2

        s_dense = float(dense_norm.get(idx, 0.0))
        s_bm25 = float(bm25_norm.get(idx, 0.0))
        score = (w_dense * s_dense) + (w_bm25 * s_bm25)

        score = score + _acute_bonus(txt) - _chronic_penalty(txt)
        score -= _clinical_mismatch_penalty(title, txt, q_low)

        acute_hits = _count_hits(txt, ACUTE_HINT_TERMS)
        if query_is_acute and acute_hits == 0:
            score -= 0.35

        has_motor = any(t in txt_low for t in MOTOR_TERMS)
        has_language = any(t in txt_low for t in LANGUAGE_TERMS)
        has_posterior = any(t in txt_low for t in POSTERIOR_TERMS)
        has_hemorrhage = any(t in txt_low for t in HEMORRHAGE_TERMS)

        requested_focus = query_has_motor or query_has_language or query_has_posterior or query_has_hemorrhage
        if requested_focus and not (has_motor or has_language or has_posterior or has_hemorrhage):
            continue

        if query_has_motor and has_motor:
            score += 0.15
        if query_has_language and has_language:
            score += 0.15
        if query_has_posterior and has_posterior:
            score += 0.20
        if query_has_hemorrhage and has_hemorrhage:
            score += 0.20

        if query_has_motor and query_has_language:
            if has_motor and has_language:
                score += 0.35
            else:
                score += 0.05

        asks_complication = any(t in q_low for t in COMPLICATION_TOPICS)
        if (query_has_motor or query_has_language) and (not asks_complication):
            if (
                any(t in txt_low for t in COMPLICATION_TOPICS)
                or any(t in title_low for t in COMPLICATION_TOPICS)
            ):
                score -= 0.30

        results.append({
            "title": title,
            "source": meta.get("source", ""),
            "pmid": pmid,
            "doi": doi,
            "text": txt,
            "snippet": safe_truncate(txt, 350),
            "score": float(score),
            "dense_score": float(s_dense),
            "bm25_score": float(s_bm25),
            "base_hybrid_score": float((w_dense * s_dense) + (w_bm25 * s_bm25)),
            "hybrid_score": float(score),
            "from_dense": bool(idx in dense_norm),
            "from_bm25": bool(idx in bm25_norm),
            "from_kg": False,
            "retrieval_channel": "hybrid",
            "retrieval_views": [name for name, enabled in (("dense", idx in dense_norm), ("bm25", idx in bm25_norm)) if enabled],
            "chunk_id": meta.get("chunk_id", meta.get("id", int(idx))),
            "_has_motor": has_motor,
            "_has_language": has_language,
            "_has_posterior": has_posterior,
            "_has_hemorrhage": has_hemorrhage,
            "clinical_signals": summarize_signals(clinical_signals),
            "hybrid_feedback": hybrid_feedback_info,
            "_from_graph": False,
        })

    # ========== إضافة نتائج Knowledge Graph ==========
    # Reuse KG results collected for hybrid feedback when available. If not
    # collected earlier, query KG now with the original case. When hybrid
    # feedback succeeds, optionally run a second-stage KG search with the
    # evidence-derived feedback query and merge both pools.
    kg_raw_limit = _kg_raw_top_k(k)
    if not graph_results:
        graph_results = _search_kg_fresh(q, top_k=kg_raw_limit, debug=debug)

    if (
        USE_KG
        and hybrid_feedback_used
        and feedback_query
        and feedback_query != q
        and bool(HYB_FB_CFG.get("second_stage_kg", True))
    ):
        kg_feedback_results = _search_kg_fresh(feedback_query, top_k=kg_raw_limit, debug=debug)
        if kg_feedback_results:
            merged_kg: Dict[str, Dict] = {}
            for item in list(graph_results) + list(kg_feedback_results):
                key = str(item.get("chunk_id") or item.get("pmid") or item.get("title") or id(item))
                prev = merged_kg.get(key)
                if prev is None:
                    merged_kg[key] = item
                    continue
                old_score = float(prev.get("graph_score", 0.0) or 0.0)
                new_score = float(item.get("graph_score", 0.0) or 0.0)
                if new_score > old_score:
                    merged_kg[key] = item
            graph_results = list(merged_kg.values())

    if debug:
        print(f"KG raw results count = {len(graph_results)}")
        if graph_results:
            preview = [
                {
                    "pmid": g.get("pmid"),
                    "title": g.get("title"),
                    "disease": g.get("disease"),
                    "category": g.get("category"),
                    "article_type": g.get("article_type"),
                    "graph_score": g.get("graph_score"),
                }
                for g in graph_results[:5]
            ]
            print("KG raw preview =", preview)

    kg_added = 0
    kg_missing_meta = 0

    if USE_KG and graph_results and KG_CFG.get("use_hybrid_search", True):
        raw_graph_scores = [float(g.get("graph_score", 0.0) or 0.0) for g in graph_results]
        norm_graph_scores = _minmax_norm_scores(raw_graph_scores)

        for g, norm_g in zip(graph_results, norm_graph_scores):
            pmid = g.get("pmid")
            if pmid is None:
                if debug:
                    print("KG skip: result without PMID")
                continue

            support_meta = _best_meta_for_pmid(str(pmid), q_low)
            if support_meta is None:
                kg_missing_meta += 1
                if debug:
                    print(f"KG skip: no support_meta for PMID {pmid} | title={g.get('title')}")
                continue

            txt = support_meta.get("text", "") or ""
            txt_low = txt.lower()
            title = g.get("title") or support_meta.get("title", "") or ""
            title_low = title.lower()
            if _should_skip_clinical_mismatch(title, txt, q_low):
                if debug:
                    print(
                        f"KG skip: clinical age mismatch | PMID {pmid} | title={title}"
                    )
                continue

            graph_score = float(g.get("graph_score", 0.0) or 0.0)
            category = str(g.get("category", "") or "").lower()
            article_type = str(g.get("article_type", "") or "").lower()

            vascular_relevance = float(g.get("vascular_relevance", 0.0) or 0.0)
            posterior_relevance = float(g.get("posterior_relevance", 0.0) or 0.0)
            hemorrhage_relevance = float(g.get("hemorrhage_relevance", 0.0) or 0.0)
            emergency_relevance = float(g.get("emergency_relevance", 0.0) or 0.0)
            chronic_penalty = float(g.get("chronic_penalty", 0.0) or 0.0)
            article_noise_penalty = float(g.get("article_noise_penalty", 0.0) or 0.0)

            score = w_graph * float(norm_g)
            score -= _clinical_mismatch_penalty(title, txt, q_low)

            # KG clinical boost
            if query_is_acute:
                score += 0.20 * emergency_relevance
                score += 0.18 * vascular_relevance
                score -= 0.20 * chronic_penalty
                score -= 0.15 * article_noise_penalty

            if query_has_posterior:
                score += 0.30 * posterior_relevance
                if category == "posterior_vascular" or "posterior" in article_type:
                    score += 0.25

            if query_has_hemorrhage:
                score += 0.35 * hemorrhage_relevance
                if category == "hemorrhagic" or "hemorrhage" in article_type:
                    score += 0.25

            if query_has_motor or query_has_language:
                if category == "vascular":
                    score += 0.25
                if any(t in title_low for t in [
                    "acute ischemic stroke", "acute ischaemic stroke",
                    "cerebral infarction", "ischemia", "ischaemia",
                    "transient ischemic attack", "tpa", "thrombolysis"
                ]):
                    score += 0.20

            # focal بدون posterior: خفف posterior-specific KG
            if (query_has_motor or query_has_language) and not query_has_posterior:
                if category == "posterior_vascular" or posterior_relevance >= 0.8 or "posterior" in article_type:
                    score -= 0.25

            if query_has_motor and any(t in txt_low for t in MOTOR_TERMS):
                score += 0.10
            if query_has_language and any(t in txt_low for t in LANGUAGE_TERMS):
                score += 0.10
            if query_has_posterior and any(t in txt_low for t in POSTERIOR_TERMS):
                score += 0.15
            if query_has_hemorrhage and any(t in txt_low for t in HEMORRHAGE_TERMS):
                score += 0.15
            if query_is_acute:
                score += _acute_bonus(txt) * 0.5

            results.append({
                "title": title,
                "source": g.get("source", "GraphKB"),
                "pmid": pmid,
                "doi": support_meta.get("doi"),
                "text": txt,
                "snippet": safe_truncate(txt, 350),
                "score": float(score),
                "hybrid_score": float(score),
                "dense_score": None,
                "bm25_score": None,
                "from_dense": False,
                "from_bm25": False,
                "from_kg": True,
                "retrieval_channel": "kg",
                "retrieval_views": ["kg"],
                "chunk_id": support_meta.get("chunk_id", support_meta.get("id")),
                "graph_score": graph_score,
                "kg_paths": list(g.get("kg_paths") or []),
                "graph_disease": g.get("disease"),
                "graph_matched_symptoms": g.get("matched_symptoms", []),
                "graph_category": g.get("category"),
                "graph_article_type": g.get("article_type"),
                "graph_acuity": g.get("acuity"),
                "graph_query_flags": g.get("query_flags"),
                "_has_motor": any(t in txt_low for t in MOTOR_TERMS),
                "_has_language": any(t in txt_low for t in LANGUAGE_TERMS),
                "_has_posterior": any(t in txt_low for t in POSTERIOR_TERMS),
                "_has_hemorrhage": any(t in txt_low for t in HEMORRHAGE_TERMS),
                "clinical_signals": summarize_signals(clinical_signals),
                "hybrid_feedback": hybrid_feedback_info,
                "_from_graph": True,
            })
            kg_added += 1

            if debug:
                print(
                    f"KG add: PMID {pmid} | graph_score={g.get('graph_score')} | "
                    f"final_score={score:.4f} | category={category} | title={title}"
                )
    # =================================================

    if debug:
        print(f"KG linked results added = {kg_added}")
        print(f"KG missing support_meta = {kg_missing_meta}")

    dedup: Dict[str, Dict] = {}
    for item in results:
        chunk_id = item.get("chunk_id")
        pmid = item.get("pmid")
        title = item.get("title", "")
        key = str(chunk_id) if chunk_id else f"pmid::{pmid}::{title}"

        prev = dedup.get(key)
        if prev is None:
            dedup[key] = item
            continue

        prev_score = float(prev.get("score", 0.0) or 0.0)
        new_score = float(item.get("score", 0.0) or 0.0)

        if new_score > prev_score:
            merged = prev.copy()
            merged.update(item)
            dedup[key] = merged
        else:
            if item.get("_from_graph") and not prev.get("_from_graph"):
                prev["graph_score"] = item.get("graph_score")
                prev["graph_disease"] = item.get("graph_disease")
                prev["graph_matched_symptoms"] = item.get("graph_matched_symptoms", [])
                prev["graph_category"] = item.get("graph_category")
                prev["graph_article_type"] = item.get("graph_article_type")
                prev["graph_acuity"] = item.get("graph_acuity")
                prev["graph_query_flags"] = item.get("graph_query_flags")
                prev["_from_graph"] = True

    results = list(dedup.values())
    results.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)

    if debug:
        top_graph = [
            {
                "title": r.get("title"),
                "pmid": r.get("pmid"),
                "graph_score": r.get("graph_score"),
                "graph_disease": r.get("graph_disease"),
                "graph_category": r.get("graph_category"),
                "score": r.get("score"),
                "source": r.get("source"),
            }
            for r in results[:10]
            if r.get("graph_score") is not None or r.get("source") == "GraphKB"
        ]
        print("Top graph-contributing results =", top_graph)

        # =================================================
    # Focus-aware final ordering + KG quota
    # =================================================
    if query_has_motor and query_has_language:
        both = [r for r in results if r.get("_has_motor") and r.get("_has_language")]
        motor_only = [r for r in results if r.get("_has_motor") and not r.get("_has_language")]
        lang_only = [r for r in results if r.get("_has_language") and not r.get("_has_motor")]

        focus_ordered = []
        if both:
            focus_ordered = both + motor_only + lang_only
        else:
            i = 0
            while i < max(len(motor_only), len(lang_only)):
                if i < len(motor_only):
                    focus_ordered.append(motor_only[i])
                if i < len(lang_only):
                    focus_ordered.append(lang_only[i])
                i += 1

        used = {
            str(r.get("chunk_id")) if r.get("chunk_id") else f"pmid::{r.get('pmid')}::{r.get('title', '')}"
            for r in focus_ordered
        }
        focus_ordered += [
            r for r in results
            if (
                str(r.get("chunk_id")) if r.get("chunk_id") else f"pmid::{r.get('pmid')}::{r.get('title', '')}"
            ) not in used
        ]

    elif query_has_posterior:
        posterior_first = [r for r in results if r.get("_has_posterior")]
        others = [r for r in results if not r.get("_has_posterior")]
        focus_ordered = posterior_first + others

    elif query_has_hemorrhage:
        hemorrhage_first = [r for r in results if r.get("_has_hemorrhage")]
        others = [r for r in results if not r.get("_has_hemorrhage")]
        focus_ordered = hemorrhage_first + others

    else:
        focus_ordered = results

    max_kg_final = _kg_return_cap(k)

    final = []
    kg_count = 0
    seen_final = set()

    for r in focus_ordered:
        key = str(r.get("chunk_id")) if r.get("chunk_id") else f"pmid::{r.get('pmid')}::{r.get('title', '')}"
        if key in seen_final:
            continue

        is_kg = _is_kg_item(r)

        if is_kg:
            if kg_count >= max_kg_final:
                continue
            kg_count += 1

        final.append(r)
        seen_final.add(key)

        if len(final) >= k:
            break

    for r in final:
        r.pop("_has_motor", None)
        r.pop("_has_language", None)
        r.pop("_has_posterior", None)
        r.pop("_has_hemorrhage", None)
        r.pop("_from_graph", None)

    return final


# alias
search = hybrid_search