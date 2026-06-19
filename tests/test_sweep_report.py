import json
import importlib.util
import shutil
import unittest
import uuid
from pathlib import Path

from scripts import run_artifact_sweep as sweep

HAS_REPORT_DEPS = all(
    importlib.util.find_spec(name) is not None for name in ("matplotlib", "numpy", "pandas", "seaborn")
)
if HAS_REPORT_DEPS:
    import pandas as pd

    from scripts import build_sweep_report as report
else:
    pd = None
    report = None


def _tempdir():
    base = Path(__file__).resolve().parent / "_tmp_sweep"
    base.mkdir(parents=True, exist_ok=True)
    path = base / ("t" + uuid.uuid4().hex)
    path.mkdir(parents=False, exist_ok=False)

    class _Ctx:
        def __enter__(self):
            return path

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(path, ignore_errors=True)

    return _Ctx()


class TestDoseLadderSweepReport(unittest.TestCase):
    def test_write_dose_ladder_matrix_uses_baseline_for_zero_dose(self) -> None:
        with _tempdir() as td:
            matrix_path = sweep.write_dose_ladder_matrix(
                td,
                {"id": "reality", "text": "What is reality?"},
                dose_values=[0, 64, 128],
                dose_conditions=["assoc", "facts"],
            )
            data = json.loads(matrix_path.read_text(encoding="utf-8"))
            self.assertTrue(data["include_baseline"])
            names = [item["name"] for item in data["items"]]
            self.assertEqual(names, ["assoc_dose_64", "assoc_dose_128", "facts_dose_64", "facts_dose_128"])
            self.assertEqual(data["items"][0]["cfg"]["scaffold_condition"], "assoc")
            self.assertEqual(data["items"][0]["cfg"]["scaffold_token_target"], 64)

    def test_extracts_dose_and_attractor_metrics_from_pack_json(self) -> None:
        if report is None or pd is None:
            self.skipTest("optional report-analysis dependencies are not installed")

        with _tempdir() as td:
            result_path = td / "gemini_reality.json"
            trace_path = td / "gemini_reality.trace.jsonl"
            result = {
                "kind": "lab_matrix",
                "provider": "custom",
                "model": "gemini-3.1-pro-preview",
                "query": "What is reality interface?",
                "items": [
                    {
                        "name": "baseline",
                        "kind": "baseline",
                        "answer": "Reality is an interface for choice and constraint.",
                        "metrics": {"answer_chars": 51, "elapsed_s": 1.0},
                    },
                    {
                        "name": "assoc_dose_64",
                        "kind": "ponder",
                        "control": "none",
                        "answer": "Reality is an interface path where choice and constraint recurse.",
                        "cfg_overrides": {"scaffold_condition": "assoc", "scaffold_token_target": 64},
                        "metrics": {"answer_chars": 66, "records": 1, "elapsed_s": 2.0},
                        "records": [{"ponder_question": "Where does the interface bend?", "ponder_log": "choice path constraint"}],
                        "extras": {"scaffold": {"condition": "assoc", "target_tokens": 64}},
                        "comparison": {
                            "diff_ratio": 0.2,
                            "semantic": {"answer_cosine": 0.7, "query_alignment_delta": 0.18},
                            "stance": {"shift_score": 0.1},
                            "spatial_metaphor": {"logs": {"density_per_1k_chars": 3.0}},
                            "token_budget": {"delta": {"api_total_tokens": 120, "external_scaffold_tokens_est": 64}},
                        },
                    },
                ],
            }
            result_path.write_text(json.dumps(result), encoding="utf-8")
            trace_path.write_text(
                json.dumps(
                    {
                        "event": "scaffold_conditioned",
                        "pack_item": "assoc_dose_64",
                        "text_preview": "choice path constraint recurse",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = {
                "runs": [
                    {
                        "provider": "gemini",
                        "query_id": "reality",
                        "query": "What is reality interface?",
                        "json_out": str(result_path),
                        "trace_out": str(trace_path),
                        "matrix_report_out": str(td / "matrix.html"),
                        "trace_report_out": str(td / "trace.html"),
                        "stdout": str(td / "stdout.txt"),
                        "stderr": str(td / "stderr.txt"),
                        "ok": True,
                        "elapsed_s": 3.0,
                    }
                ]
            }
            (td / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            run_rows, item_rows = report.extract_rows(td)
            self.assertEqual(item_rows[0]["scaffold_condition"], "assoc")
            self.assertEqual(item_rows[0]["scaffold_dose"], 64)

            run_df = pd.DataFrame(run_rows)
            item_df = pd.DataFrame(item_rows)
            attractor_rows = report.build_attractor_rows(run_df, item_df)
            self.assertEqual(attractor_rows[0]["prior_item"], "baseline")
            self.assertGreater(attractor_rows[0]["origin_drift"], 0)
            self.assertGreater(attractor_rows[0]["coil_index"], 0)

            dose_rows = report.build_dose_response_rows(run_df, item_df, attractor_rows)
            dose_values = sorted(row["scaffold_dose"] for row in dose_rows)
            self.assertEqual(dose_values, [0, 64])
            self.assertTrue(any(row["is_baseline_reference"] for row in dose_rows))


if __name__ == "__main__":
    unittest.main()
