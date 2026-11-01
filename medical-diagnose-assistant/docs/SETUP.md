# Setup and required artifacts

The cleaned source restores the package layout expected by existing imports. It is not a self-contained deployment. Always start commands in the repository root because configuration and data paths are relative to it.

## 1. Python and dependencies

Install `requirements.txt` in an isolated Python 3.10+ environment. No version lock was supplied or established during cleanup. In particular, resolve spaCy/scispaCy compatibility and the original scikit-learn model version before treating the environment as reproducible.

The graph builder loads `en_ner_bc5cdr_md` at import time. Install a release compatible with your spaCy/scispaCy versions; it is a separate model distribution, not included in the ZIP. Graph retrieval may fall back or fail if this model is absent.

## 2. Corpus and indexes

Supply your authorized PubMed subset at `data/pubmed_subset/pubmed_subset.jsonl`. Each line is a JSON object with `title`, `text`, and `pmid` fields. The chunker filters stroke-related records and skips short text. No original corpus or download script was included.

```powershell
python -m app.rag.add_stroke_chunks
python -m app.rag.build_index
python -m app.rag.build_bm25
```

The chunking script deletes existing `app/kb/chunks/*.json` before rebuilding. Preserve any manually curated chunks before using it. Alternatively, supply existing chunk JSON files with nonempty `text` and useful source metadata such as `title`, `pmid`, and `chunk_id` directly in that directory.

Expected generated files:

- `app/index/faiss_hnsw.index`
- `app/index/metas.json`
- `app/index/bm25.joblib`

Use the same embedding model when building and querying indexes. Build scripts download model weights when needed. For lower-memory systems, set `MEDRAG_EMB_BATCH` to a smaller value. `MEDRAG_ALLOW_CPU=1` enables the pipeline's CPU embedding fallback; `.env` supplies this after the LLM adapter loads.

## 3. Neo4j

Provide a running Neo4j instance reachable at the URI in `config.yaml`. The source defaults to local Bolt on port 7687 with `neo4j` / `password`; these are development placeholders, not a deployed credential setup. The graph builder and summary helper also contain connection defaults. Align these with your local configuration before building or querying a graph.

```powershell
python -m app.rag.build_graph
python -m app.rag.kg_summary
```

The builder writes graph data from the chunk directory. Do not point it at a shared database without reviewing its implementation and intended data changes. No database was modified during cleanup.

## 4. LLM backend

Copy `.env.example` to `.env` and fill `HF_TOKEN` for the configured `hf_api` path. Confirm that the configured model is supported by your chosen provider. Generation is conditional; an API credential does not guarantee that every request uses an LLM.

For OpenAI-compatible providers, install `requirements-optional.txt` and set the relevant provider and API key. Local GGUF execution additionally needs a hardware-compatible `llama-cpp-python` installation and model file. Backend configuration is implemented in `app/models/llm.py`; no provider connection was tested during cleanup.

## 5. Risk model and interface

The model resides at `models/stroke_rf_pipeline.joblib`, matching both the adapter and UI. Only load serialized models from trusted sources. Training controls were removed because the archive did not include `train_stroke_rf.py` or its dataset. Prediction remains available.

Expected features: `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`. See `app/models/stroke_risk.py` for documented categories. Leave the patient JSON empty to use the RAG-only path.

Launch with `python -m streamlit run app_streamlit.py` after prerequisites are ready. Pipeline imports occur at startup, so the index-building button is not a substitute for initial command-line setup.
