# app/rag/pipeline.py
from __future__ import annotations

from typing import List, Dict, Optional
import os
import re
import yaml
import numpy as np

from app.rag.prompts import SYSTEM_PROMPT as BASE_SYSTEM
from app.rag.prompts import USER_PROMPT_TEMPLATE
from app.models.llm import generate_answer
from app.rag.utils_text import safe_truncate

# ✅ موديل السكتة (اختياري)
try:
    from app.models.stroke_risk import predict_stroke_risk
    _HAS_STROKE_MODEL = True
except Exception:
    predict_stroke_risk = None
    _HAS_STROKE_MODEL = False

# المسترجع المحلي اختياري
try:
    from app.rag.retriever import hybrid_search
    _HAS_LOCAL = True
except Exception:
    _HAS_LOCAL = False

# قراءة الإعدادات
with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f) or {}

# Embeddings (GPU-only by default)
try:
    import torch
    from sentence_transformers import SentenceTransformer as _ST

    _ALLOW_CPU = os.getenv("MEDRAG_ALLOW_CPU", "0").strip() == "1"
    if (not torch.cuda.is_available()) and (not _ALLOW_CPU):
        raise RuntimeError("CUDA غير متاح — هذا المشروع مضبوط ليعمل على GPU فقط. "
                           "إذا تريد CPU مؤقتاً: set MEDRAG_ALLOW_CPU=1")

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    EMB = _ST((CFG.get("models") or {}).get("embeddings", "sentence-transformers/all-MiniLM-L6-v2"), device=_device)
except Exception:
    EMB = None


def _encode(texts: List[str]) -> np.ndarray:
    if EMB is None:
        raise RuntimeError("SentenceTransformer غير متاح.")
    return EMB.encode(texts, convert_to_numpy=True)


def _dense_rank(query: str, docs: List[Dict], top_k: int = 8) -> List[Dict]:
    try:
        if EMB is None or not docs:
            for d in docs:
                d["score"] = float(d.get("score", 0.0) or 0.0)
            return docs[:top_k]

        qv = _encode([query])[0]
        qv = qv / (np.linalg.norm(qv) + 1e-12)

        D = _encode([d["text"] for d in docs])
        D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)

        sims = (D @ qv).tolist()
        for i, s in enumerate(sims):
            docs[i]["score"] = float(s)

        docs.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        return docs[:top_k]
    except Exception:
        for d in docs:
            d["score"] = float(d.get("score", 0.0) or 0.0)
        return docs[:top_k]


def _build_context(docs: List[Dict]) -> str:
    lines: List[str] = []
    max_total_chars = 2000
    current_len = 0

    for i, d in enumerate(docs, start=1):
        cite = f"[PMID:{d.get('pmid')}]" if d.get("pmid") else ""
        src = d.get("url") or d.get("source", "")
        title = (d.get("title") or "").strip()
        text = d.get("text", "") or ""

        snippet = safe_truncate(text, 250)

        header_parts = [f"[{i}]"]
        if cite:
            header_parts.append(cite)
        if title:
            header_parts.append(title)

        header = " ".join(header_parts).strip()

        block_parts = [header, snippet]
        if src:
            block_parts.append(f"المصدر: {src}")

        block = "\n".join(block_parts)

        if current_len + len(block) > max_total_chars:
            break

        lines.append(block)
        current_len += len(block)

    return "\n\n".join(lines)


def _cleanup_llm_answer(raw: str) -> str:
    if not raw:
        return "Evidence is insufficient"
    return raw.strip()


def _best_pmid_citation(docs: List[Dict]) -> str:
    for d in docs:
        pmid = d.get("pmid")
        if pmid:
            return f"[PMID:{pmid}]"
    return "[PMID:unknown]"


def _postprocess_local_with_citations(raw: str, docs_for_llm: List[Dict]) -> str:
    cite = _best_pmid_citation(docs_for_llm)
    raw_text = (raw or "").replace("\r\n", "\n").strip()

    # لو فعلاً قالها لحالها
    if raw_text == "Evidence is insufficient":
        return "Evidence is insufficient"

    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]

    bullets = []
    for ln in lines:
        if ln.startswith(("-", "*")) or re.match(r"^\d+[\)\.\-]\s+", ln):
            bullets.append(re.sub(r"^\d+[\)\.\-]\s+", "", ln).strip())

    if not bullets:
        text = " ".join(lines)
        sents = [s.strip() for s in re.split(r"[\.!\?]\s+", text) if s.strip()]
        bullets = sents[:3]

    if not bullets:
        return "Evidence is insufficient"

    bullets = bullets[:3]
    bullets = [b[:220].rstrip() + ("…" if len(b) > 220 else "") for b in bullets]

    fixed = []
    for b in bullets:
        if "[PMID:" in b:
            fixed.append(b)
        else:
            fixed.append(f"{b} {cite}")

    strength = "ضعيف" if len(docs_for_llm) <= 1 else "متوسط"
    return (
        "1) ملخص الأدلة:\n"
        + "\n".join(f"- {b}" for b in fixed)
        + "\n\n2) تعليق على قوة الدليل:\n"
        + f"- قوة الدليل {strength} لأنها مبنية على {len(docs_for_llm)} مقطع(مقاطع)."
    )


def _score_to_confidence(score: float, has_docs: bool) -> str:
    if not has_docs:
        return "منخفض"
    # ملاحظة: score ممكن يكون cosine أو score من hybrid_search
    if score >= 0.55:
        return "مرتفع"
    if score >= 0.35:
        return "متوسط"
    return "منخفض"


def get_answer(
    case_text: str,
    top_k: int | None = None,
    debug: bool = False,
    stroke_features: Optional[Dict] = None,
) -> Dict:
    # top_k آمن حتى لو كان string غريب
    raw_k = top_k if top_k is not None else (CFG.get("retrieval") or {}).get("top_k_merged", 6)
    try:
        k = int(str(raw_k).replace("–", "-").split("-")[0].strip())
    except Exception:
        k = 6

    pm_cfg = CFG.get("pubmed", {}) or {}
    merged_docs: List[Dict] = []

    # Local KB
    if pm_cfg.get("merge_with_local_kb", True) and _HAS_LOCAL:
        try:
            local = hybrid_search(case_text, top_k=k)
            for d in local:
                merged_docs.append({
                    "pmid": d.get("pmid"),
                    "doi": d.get("doi"),
                    "title": d.get("title", ""),
                    "text": d.get("snippet", ""),
                    "url": d.get("source", ""),
                    "journal": None,
                    "year": None,
                    "score": float(d.get("score", 0.0) or 0.0),
                })
        except Exception:
            pass

    if merged_docs:
        merged_docs.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)

    # docs for llm
    max_docs_for_llm = int((CFG.get("retrieval") or {}).get("max_docs_for_llm", 3))
    docs_for_llm = merged_docs[:max_docs_for_llm]

    context_block = _build_context(docs_for_llm) if docs_for_llm else "لا توجد مقاطع كافية."
    max_ctx_chars = int((CFG.get("retrieval") or {}).get("max_context_chars", 1200))
    context_block = safe_truncate(context_block, max_ctx_chars)

    models_cfg = CFG.get("models", {}) or {}
    provider = models_cfg.get("llm_provider", "local")
    model_name = models_cfg.get("llm_model", "Qwen/Qwen2-1.5B-Instruct")

    # -------- Stroke risk (optional) --------
    stroke_risk = None
    risk_block = ""
    if stroke_features is not None and _HAS_STROKE_MODEL and predict_stroke_risk is not None:
        try:
            stroke_risk = predict_stroke_risk(stroke_features)
            prob_stroke = stroke_risk.get("prob_stroke")
            thr = stroke_risk.get("threshold")
            label = stroke_risk.get("risk_label")
            risk_block = (
                "\n\n[تنبؤ خطر السكتة (نموذج ML)]\n"
                f"- prob_stroke: {prob_stroke}\n"
                f"- risk_label: {label}\n"
                f"- threshold: {thr}\n"
                "ملاحظة: هذا تقدير إحصائي لدعم القرار (ليس تشخيصاً سريرياً).\n"
            )
        except Exception:
            stroke_risk = None
            risk_block = "\n\n[تنبؤ خطر السكتة (نموذج ML)] غير متاح.\n"

    system_for_llm = BASE_SYSTEM.strip()

    # ✅ نطلب العربية، مع الحفاظ على fallback الإنجليزي الحرفي
    user_for_llm = (
        USER_PROMPT_TEMPLATE.format(case=case_text).strip()
        + (risk_block if risk_block else "")
        + "\n\nContext (evidence):\n"
        + context_block
        + "\n\nاكتب بالعربية فقط (ممنوع أي كلمات إنجليزية/صينية/فرنسية). إذا الأدلة غير كافية قل حرفياً: Evidence is insufficient.\n"
    )

    llm_raw = generate_answer(
        system_for_llm,
        user_for_llm,
        provider=provider,
        model_name=model_name,
    )
    raw_text = str(llm_raw)

    if debug:
        print("RAW ...")
        print(raw_text[:1200])

    if provider == "local":
        answer = _postprocess_local_with_citations(raw_text, docs_for_llm)
    else:
        answer = _cleanup_llm_answer(raw_text)

    sources = []
    for d in merged_docs[:k]:
        sources.append({
            "title": d.get("title", ""),
            "source": d.get("url", ""),
            "pmid": d.get("pmid"),
            "doi": d.get("doi"),
            "score": float(d.get("score", 0.0) or 0.0),
            "snippet": safe_truncate(d.get("text", ""), 300),
        })

    top_score = float(merged_docs[0].get("score", 0.0) or 0.0) if merged_docs else 0.0
    confidence = _score_to_confidence(top_score, bool(merged_docs))

    out = {
        "answer": answer,
        "confidence": confidence,
        "top_score": top_score,
        "sources": sources,
    }
    if stroke_risk is not None:
        out["stroke_risk"] = stroke_risk
    if debug:
        out["debug_evidence_for_llm"] = docs_for_llm
        out["debug_user_prompt"] = user_for_llm
    return out


def get_stroke_answer(
    case_text: str,
    stroke_features: Dict,
    top_k: int | None = None,
    debug: bool = False,
) -> Dict:
    """
    دمج:
      - RAG summary (get_answer)
      - Stroke risk model (predict_stroke_risk)
    """
    if not _HAS_STROKE_MODEL or predict_stroke_risk is None:
        raise RuntimeError("stroke_risk model غير متاح. تأكد من وجود app/models/stroke_risk.py وملف الموديل joblib.")

    # الأفضل: نحسبها ضمن get_answer أيضاً لتدخل بالـprompt
    rag_result = get_answer(case_text, top_k=top_k, debug=debug, stroke_features=stroke_features)

    # تأكيد وجود stroke_risk في الخرج
    if "stroke_risk" not in rag_result:
        rag_result["stroke_risk"] = predict_stroke_risk(stroke_features)

    return rag_result
