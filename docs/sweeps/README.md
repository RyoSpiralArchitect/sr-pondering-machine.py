# Sweep Reports

Curated HTML report bundles for Claude/Gemini `sr_pondering_machine.py` artifact sweeps.

GitHub's file browser shows checked-in `.html` files as source. Use the rendered links below for the interactive GitHub Pages reports, and the source links when you want to inspect the committed files.

- [2026-06-17 standard sweep](https://ryospiralarchitect.github.io/sr-pondering-machine.py/sweeps/2026-06-17-standard/analysis/report.html) ([source](2026-06-17-standard/analysis/report.html)): Claude Haiku and Gemini 3.5 Flash exploratory run.
- [2026-06-18 strong deep sweep](https://ryospiralarchitect.github.io/sr-pondering-machine.py/sweeps/2026-06-18-strong-deep/analysis/report.html) ([source](2026-06-18-strong-deep/analysis/report.html)): Claude Opus 4.8 with `effort=xhigh` and Gemini 3.1 Pro Preview with `reasoning_effort=high`.
- [2026-06-19 scaffold dose ladder](https://ryospiralarchitect.github.io/sr-pondering-machine.py/sweeps/2026-06-19-dose-ladder/analysis/report.html) ([source](2026-06-19-dose-ladder/analysis/report.html)): Refreshed Gemini 3.1 Pro Preview high-reasoning dose sweep across `assoc`, `random`, `facts`, and `isomorphic`; the latest run keeps the full plot bundle, PCA, raw outputs, and a strongest query-alignment row at `facts_dose_128`.
- [2026-06-19 Gemini closure contract](https://ryospiralarchitect.github.io/sr-pondering-machine.py/sweeps/2026-06-19-gemini-closure-contract/analysis/report.html) ([source](2026-06-19-gemini-closure-contract/analysis/report.html)): Gemini 3.1 Pro Preview high-reasoning sweep across final, log, and non-semantic log-skeleton closure contracts; final closure landed reliably, while log closure remained the weak point even with an `X1|...X4|...END_LOG` skeleton.
- [2026-06-19 Gemini log-phase routes](https://ryospiralarchitect.github.io/sr-pondering-machine.py/sweeps/2026-06-19-gemini-log-phase-routes/analysis/report.html) ([source](2026-06-19-gemini-log-phase-routes/analysis/report.html)): Gemini 3.1 Pro Preview sweep that isolates the ponder-log call from the final-answer call. `inherit_1024_no_rescue` failed the log skeleton in both scaffold conditions, while `low_1024_no_rescue` closed the `X1|...X4|...END_LOG` skeleton in both rows without needing rescue.

Each bundle includes the rendered report, chart PNGs, metric CSVs, PCA CSV, raw result JSON, matrix HTML, trace HTML/JSONL, and captured stdout/stderr. Dose-ladder bundles also include `dose_response_metrics.csv` and `attractor_metrics.csv`; closure-contract and log-phase-route bundles also include `closure_contract_metrics.csv`.
