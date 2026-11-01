# Cleanup and validation

## Presentation reference

The reference was [W0LF37/Movie-Recommendation-System](https://github.com/W0LF37/Movie-Recommendation-System), inspected through GitHub's README and root-contents API. Its concise introduction, implementation table, Mermaid flow, repository tree, separate technical documentation, and explicit reproducibility limits informed this presentation. Its notebook-specific folders and numerical results were not copied into this project.

## Changes

- Restored `app/rag` and `app/models` to match existing absolute and relative imports; added package markers.
- Moved the pretrained model to `models/stroke_rf_pipeline.joblib`, matching the adapter's existing path; corrected the UI path to match.
- Moved both original PDFs to `docs/reports` without changing their contents.
- Removed only the stray one-byte file `,,,`. No core code was deleted.
- Replaced repetitive bilingual README sections and unsupported "stable / working end-to-end" claims with a source-grounded overview, setup, architecture and limitations.
- Added `.gitignore`, `.env.example`, an inventory of direct runtime dependencies and optional provider requirements. Removed duplicated dotenv and unused web-framework/vector-store packages from the dependency list; no source imports those removed packages.
- Renamed UI branding to Med-RAG; fixed blank or malformed patient JSON handling in the question flow.
- Removed the nonfunctional training controls that imported the missing `train_stroke_rf` module; retained pretrained prediction and clearly documented missing training assets.

## Validation

All supplied Python files pass syntax parsing after reorganization. A local-import audit resolves all supplied `app.*` modules; the original optional `clinical_subtype_resolver` remains absent and guarded by fallback logic. The removed trainer import no longer exists.

Byte comparisons and SHA-256 hashes verify preservation of all non-UI Python source files, both PDF reports, and the serialized model. See `cleanup-manifest.json` for exact paths and hashes. Focused UI checks cover the optional patient-JSON branch and model path. Lightweight clinical-context modules are imported without loading external ML models to verify the restored package structure; this does not establish clinical accuracy.

Full application execution, provider calls, model deserialization/inference, graph construction, dependency installation, and clinical evaluation were not performed. Required corpus, indexes, database, scientific NLP model, and original model environment are unavailable. The requirements files are not a tested environment lock.
