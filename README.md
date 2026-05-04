Med-RAG (GraphRAG Enhanced)
Medical Retrieval-Augmented Generation System
=============================================

## OVERVIEW

Med-RAG is a medical question-answering and decision support system
based on Retrieval-Augmented Generation (RAG), enhanced with a
Knowledge Graph (GraphRAG).

The system combines:

* Hybrid medical document retrieval
* Knowledge Graph reasoning
* Rule-based clinical logic

to generate evidence-based, grounded medical insights.

The current focus is **stroke triage and neurological risk detection**.

## KEY FEATURES

* Hybrid Retrieval (FAISS + BM25 + Knowledge Graph)
* Knowledge Graph integration (Neo4j)
* Evidence-grounded medical outputs (PMIDs)
* Rule-based clinical decision engine
* Posterior stroke detection logic
* Minimal reliance on LLMs (fallback only)
* Debug-friendly pipeline (transparent scoring)

## SYSTEM ARCHITECTURE

Input (clinical text)
↓
Hybrid Retrieval (FAISS + BM25 + GraphKB)
↓
Reranking (MedCPT Cross-Encoder)
↓
Grounding Validation
↓
Rule Engine (Primary Decision)
↓
LLM (Fallback only if needed)
↓
Output (Diagnosis + Evidence + Confidence)

## CURRENT CAPABILITIES

* Detects high-risk neurological conditions:

  * Posterior circulation stroke
  * Brainstem infarction
  * Cerebellar stroke
  * Subarachnoid hemorrhage (SAH)

* Handles clinical patterns such as:

  * Vertigo + diplopia + ataxia
  * Thunderclap headache + LOC
  * Short-duration neurological symptoms (TIA awareness)

* Uses Knowledge Graph signals to strengthen clinical reasoning

## TECH STACK

* Python
* FAISS (semantic search)
* BM25 (lexical retrieval)
* Neo4j (Knowledge Graph)
* MedCPT (reranking model)
* spaCy / scispaCy
* Docker (for KG)

## HOW TO RUN

1. Activate environment:

   .venv\Scripts\activate

2. Run full pipeline:

   python -c "from app.rag.pipeline import get_answer; print(get_answer('ongoing vertigo diplopia gait ataxia for 3 hours'))"

3. Test retrieval only:

   python -c "from app.rag.retriever import hybrid_search; print(hybrid_search('vertigo diplopia ataxia', top_k=10))"

## DEBUG & TROUBLESHOOTING

If something is not working:

1. Check Neo4j:
   docker ps

2. Ensure KG connection:
   bolt://127.0.0.1:7687

3. Verify KG usage:

   * graph_score should NOT be None
   * source should be "GraphKB"

4. Check model warnings:
   "No sentence-transformers model found"
   → install sentence-transformers if needed

## SAFETY AND SCOPE

* Informational and research use only
* No final medical diagnosis
* No treatment recommendations
* Outputs must be interpreted by professionals
* System prioritizes evidence grounding

## INTENDED USE

* Medical education
* Clinical AI research
* Stroke triage experimentation
* RAG + GraphRAG system development

## PLATFORM

* Python-based
* Runs locally
* Windows compatible
* Requires Neo4j (Docker recommended)

## CURRENT STATUS

* Hybrid retrieval: Stable
* Knowledge Graph: Integrated and active
* Pipeline: Working end-to-end
* Clinical outputs: Meaningful and grounded

System is at **MVP+ stage** (functional with ongoing optimization).

## LIMITATIONS

* Knowledge Graph still improving (semantic quality)
* MedCPT model may fallback if not fully loaded
* Temporal reasoning (TIA vs stroke) is basic
* Not production-ready for real clinical deployment

## NEXT STEPS

* Improve Knowledge Graph relevance (stroke-aware KG)
* Optimize fusion weighting (FAISS vs KG)
* Add temporal reasoning
* Expand dataset (imaging + clinical reports)
* Confidence calibration

## DISCLAIMER

This software is intended for research and educational purposes only.
It must NOT be used for real medical decision-making.

Always consult qualified healthcare professionals.

---

Med-RAG (GraphRAG Enhanced)
نظام توليد إجابات طبية معززة بالاسترجاع
=======================================

## نظرة عامة

Med-RAG هو نظام للإجابة على الأسئلة الطبية يعتمد على
تقنية التوليد المعزز بالاسترجاع (RAG) مع دعم Knowledge Graph.

يركز النظام حالياً على تحليل حالات السكتة الدماغية
واكتشاف المخاطر العصبية.

## الميزات الأساسية

* استرجاع هجين (FAISS + BM25 + Knowledge Graph)
* دمج Knowledge Graph (Neo4j)
* نتائج مدعومة بالأدلة (PMIDs)
* منطق طبي قائم على القواعد
* كشف السكتات الخلفية (Posterior Stroke)
* تقليل الاعتماد على النماذج اللغوية

## آلية العمل

نص الحالة الطبية
↓
استرجاع هجين
↓
إعادة ترتيب النتائج
↓
التحقق من الأدلة
↓
محرك القواعد
↓
ناتج (تشخيص + أدلة + ثقة)

## الحالة الحالية

* النظام يعمل بشكل كامل
* Knowledge Graph مدمج ويؤثر على النتائج
* النتائج ذات معنى سريري

لكن ما زال في مرحلة التطوير والتحسين

## نطاق الاستخدام

* للتعليم والبحث فقط
* ليس بديلاً عن الطبيب
* لا يقدم تشخيص نهائي

## إخلاء مسؤولية

هذا النظام مخصص للأغراض التعليمية فقط
ولا يجب استخدامه لاتخاذ قرارات طبية فعلية
