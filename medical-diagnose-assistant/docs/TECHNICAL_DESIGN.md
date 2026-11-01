# Technical design

## Entry points

`app_streamlit.py` presents case-based questions and structured Random Forest predictions. `app/rag/pipeline.py` exposes `get_answer` and `get_stroke_answer`; the latter supplies structured risk features to the main answer path.

## Evidence processing

`clinical_context.py`, `clinical_intent.py`, and `clinical_signals.py` process clinical mentions, contextual facts, and query expansion signals. `retriever.py` combines vector and lexical retrieval; `hybrid_feedback.py` supports retrieval feedback. `graph_retriever.py` obtains graph-derived information through `build_graph.py` and Neo4j.

`agent_controller.py` and `agent_state.py` coordinate the evidence workflow. `tools.py`, `evidence_judge.py`, and `explainability.py` support evidence selection, grounding, and structured explanations. Legacy and fallback branches remain in the source; this cleanup does not change their selection logic or remove versioned overrides based on appearance alone.

## Model adapters

`app/models/medcpt_reranker.py` wraps the cross-encoder. `llm.py` implements remote and local generation backends. `stroke_risk.py` loads the bundled model lazily from the root `models` directory. The 0.05 threshold is retained from the source without endorsing its calibration.

## Build artifacts

`add_stroke_chunks.py` converts a local filtered JSONL subset to chunk JSON files. `build_index.py` builds FAISS and optionally BM25; `build_bm25.py` builds lexical artifacts from metadata. `build_graph.py` constructs the knowledge graph; `kg_summary.py` inspects it. Corpus/index/database artifacts were absent from the original ZIP.

## Engineering boundaries

All 23 non-UI Python modules retain their original bytes. The package layout restores existing imports rather than introducing a new architecture. A small `app/knowledge_graph` compatibility module exposes the same graph retriever at the alternative import path used by the original runtime. The only edited original Python file is the Streamlit interface. No clinical rules, scoring, thresholds, prompts, or model weights were changed. No benchmark results are asserted.
