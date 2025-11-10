# AI Medical Summarizer – Phase 1

This package implements the structured-data pipeline for the longitudinal visit timeline summariser. It loads patient events from the MEDS (OMOP) export, groups encounters, adds medication and lab intelligence, and exposes a FastAPI service that can return both structured JSON and a markdown narrative (Gemini 1.5 Pro if configured, otherwise a rule-based fallback).

## Project Layout

```
summarizer/
├── api.py                # FastAPI entrypoint
├── config.py             # Environment-driven settings
├── data_loader.py        # Polars-based event ingestion with caching
├── errors.py             # Custom exception types
├── medication_tracker.py # Medication start/stop/change detection
├── metrics.py            # Lightweight timing & counter recorder
├── reasoning_engine.py   # Rule-based explanations for med changes
├── summarizer.py         # Service orchestration + Gemini interface
├── timeline_builder.py   # Visit-level structured timeline builder
├── trend_analyzer.py     # Lab/vital trend computation
├── visit_grouper.py      # Visit boundary detection
└── tests/                # Pytest coverage for the critical modules
```

## Configuration

Settings are read from environment variables and validated on startup (`config.load_settings()`):

| Variable | Purpose | Default |
| --- | --- | --- |
| `EHRSHOT_DATA_GLOB` | Parquet glob for MEDS events | `<repo>/data/meds_omop_ehrshot/.../*.parquet` |
| `EHRSHOT_CODES_PATH` | Path to `codes.parquet` | `<repo>/data/meds_omop_ehrshot/.../codes.parquet` |
| `SUMMARIZER_CACHE_ENTRIES` | LRU cache size for patient events | `32` |
| `SUMMARIZER_STOP_GAP_DAYS` | Gap threshold before marking medications as stopped | `90` |
| `SUMMARY_TEMPERATURE` | Gemini generation temperature | `0.2` |
| `SUMMARY_MAX_TOKENS` | Gemini max output tokens | `3000` |
| `GEMINI_API_KEY` | API key for Gemini 1.5 Pro | _required for live LLM_ |
| `GEMINI_MODEL` | Gemini model id | `models/gemini-1.5-pro` |
| `USE_FAKE_LLM` | Force offline fallback summariser (`true/false`) | `false` |
| `SUMMARIZER_SLOW_THRESHOLD` | Warn when an operation exceeds this many seconds | `5.0` |

## Running the API

```bash
cd medi-os/services/manage-agent
./venv/Scripts/python.exe -m pip install -r requirements.txt
./venv/Scripts/uvicorn summarizer.api:app --reload
```

Endpoints:

- `GET /health` – service heartbeat
- `GET /patients?limit=20` – sample patient IDs (reads from Parquet)
- `GET /summarize/{subject_id}?visit_limit=50` – full summary JSON + markdown
- `GET /metrics` – recent timings and counters

## Tests

```bash
cd medi-os/services/manage-agent
./venv/Scripts/python.exe -m pytest summarizer/tests -q
```

Coverage includes visit grouping, medication change detection, and trend analysis.

## Notes

- Gemini integration is optional; without `GEMINI_API_KEY` the service produces a deterministic rule-based summary.
- The pipeline relies on the MEDS export (`meds_omop_ehrshot`) being available at the default location or configured via environment variables.
- Trend computation currently focuses on core cardiovascular/metabolic metrics; thresholds can be extended via `ABNORMAL_THRESHOLDS` in `trend_analyzer.py`.

