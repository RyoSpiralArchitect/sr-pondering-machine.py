# Sweep Reports

Curated HTML report bundles for Claude/Gemini `sr_pondering_machine.py` artifact sweeps.

- [2026-06-17 standard sweep](2026-06-17-standard/analysis/report.html): Claude Haiku and Gemini 3.5 Flash exploratory run.
- [2026-06-18 strong deep sweep](2026-06-18-strong-deep/analysis/report.html): Claude Opus 4.8 with `effort=xhigh` and Gemini 3.1 Pro Preview with `reasoning_effort=high`.
- [2026-06-19 scaffold dose ladder](2026-06-19-dose-ladder/analysis/report.html): Gemini 3.1 Pro Preview high-reasoning dose sweep across `assoc`, `random`, `facts`, and `isomorphic`, with dose-response charts, attractor proxy metrics, PCA, and raw outputs.

Each bundle includes the rendered report, chart PNGs, metric CSVs, PCA CSV, raw result JSON, matrix HTML, trace HTML/JSONL, and captured stdout/stderr. Dose-ladder bundles also include `dose_response_metrics.csv` and `attractor_metrics.csv`.
