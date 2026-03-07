import io
import json
import os
import shutil
import unittest
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


import sr_pondering_machine as sp
import sr_trace_report as tr


class _Chdir:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._prev: str = ""

    def __enter__(self) -> None:
        self._prev = os.getcwd()
        os.chdir(str(self._path))

    def __exit__(self, exc_type, exc, tb) -> None:
        os.chdir(self._prev)


def _tempdir():
    # NOTE: `tempfile.TemporaryDirectory()` uses `0o700` on Windows, which can
    # create directories that are not accessible in some locked-down envs.
    # Keep tests hermetic by allocating temp dirs under the repo with default
    # permissions (no explicit mode).
    base = Path(__file__).resolve().parent / "_tmp2"
    base.mkdir(parents=True, exist_ok=True)

    p = base / ("t" + uuid.uuid4().hex)
    p.mkdir(parents=False, exist_ok=False)

    class _Ctx:
        def __enter__(self):
            return str(p)

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(p, ignore_errors=True)

    return _Ctx()


def _run_main(argv: list[str]) -> tuple[str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err), patch.object(
        sp.sys, "argv", ["sr_pondering_machine.py"] + list(argv)
    ):
        sp.main()
    return out.getvalue(), err.getvalue()


class TestArtifactsAndTrace(unittest.TestCase):
    def test_write_json_dest_dash(self) -> None:
        s = io.StringIO()
        p = sp.write_json_dest("-", {"hello": 1}, stream=s)
        self.assertIsNone(p)
        self.assertIn('"hello"', s.getvalue())

    def test_write_json_dest_file_creates_parent(self) -> None:
        with _tempdir() as td:
            td_path = Path(td)
            with _Chdir(td_path):
                p = sp.write_json_dest("out.json", {"ok": True})
                self.assertIsNotNone(p)
                self.assertTrue(Path("out.json").exists())
                obj = json.loads(Path("out.json").read_text(encoding="utf-8"))
                self.assertEqual(obj.get("ok"), True)

                # Nested parent directories are created.
                p2 = sp.write_json_dest("nested/sub/out.json", {"n": 1})
                self.assertIsNotNone(p2)
                self.assertTrue(Path("nested/sub/out.json").exists())

    def test_trace_writer_safe_mkdir_parent_dot(self) -> None:
        with _tempdir() as td:
            td_path = Path(td)
            with _Chdir(td_path):
                tw = sp.TraceWriter(Path("trace.jsonl"), session_id="s", preview_chars=0)
                tw.event("hello", x=1)
                txt = Path("trace.jsonl").read_text(encoding="utf-8")
                self.assertIn('"event": "hello"', txt)

    def test_stream_trace_writer(self) -> None:
        s = io.StringIO()
        tw = sp.StreamTraceWriter(s, session_id="s", preview_chars=0, label="-")
        tw.event("hello", x=1)
        self.assertIn('"event": "hello"', s.getvalue())


class TestProbeCompare(unittest.TestCase):
    def test_build_probe_compare_tracks_rank_flips(self) -> None:
        before = [
            {"token": "alpha", "token_id": 1, "rank": 0, "prob": 0.60},
            {"token": "beta", "token_id": 2, "rank": 1, "prob": 0.30},
            {"token": "gamma", "token_id": 3, "rank": 2, "prob": 0.10},
        ]
        after = [
            {"token": "beta", "token_id": 2, "rank": 0, "prob": 0.50},
            {"token": "delta", "token_id": 4, "rank": 1, "prob": 0.30},
            {"token": "alpha", "token_id": 1, "rank": 2, "prob": 0.20},
        ]

        comp = sp.build_probe_compare(before, after, top_n=3)

        self.assertTrue(comp["top1_changed"])
        self.assertEqual(comp["overlap_count"], 2)
        self.assertEqual(comp["entered_count"], 1)
        self.assertEqual(comp["exited_count"], 1)
        self.assertEqual(comp["mover_count"], 2)
        self.assertGreater(comp["js_divergence"], 0.0)
        self.assertEqual(comp["entered"][0]["token"], "delta")
        self.assertEqual(comp["exited"][0]["token"], "gamma")
        self.assertEqual({x["token"] for x in comp["movers"]}, {"alpha", "beta"})

    def test_make_probe_compare_timeline_entry(self) -> None:
        entry = sp.make_probe_compare_timeline_entry(
            source="current_records",
            point="stage",
            record={
                "ponder_ix": 3,
                "band_label": "mid",
                "band_ponder_ix": 1,
                "hop_ix": 2,
                "pipeline_stage_ix": 0,
                "ponder_mode": "counterexample",
            },
            compare_from_base={"js_divergence": 0.12},
            compare_from_prev={"js_divergence": 0.03},
            memory_chars=120,
            prompt_chars=240,
        )
        self.assertEqual(entry["source"], "current_records")
        self.assertEqual(entry["point"], "stage")
        self.assertEqual(entry["band_label"], "mid")
        self.assertEqual(entry["hop_ix"], 2)
        self.assertEqual(entry["memory_chars"], 120)
        self.assertEqual(entry["compare_from_base"]["js_divergence"], 0.12)
        self.assertEqual(entry["compare_from_prev"]["js_divergence"], 0.03)

    def test_print_config_only_includes_probe_compare(self) -> None:
        out, _err = _run_main(
            [
                "--backend",
                "openai_compat",
                "--model",
                "dummy",
                "--query",
                "q",
                "--probe_compare",
                "--probe_compare_stages",
                "--probe_compare_top_n",
                "17",
                "--print_config_only",
            ]
        )
        obj = json.loads(out)
        cfg = obj.get("cfg") or {}
        self.assertEqual(cfg.get("probe_compare"), True)
        self.assertEqual(cfg.get("probe_compare_stages"), True)
        self.assertEqual(cfg.get("probe_compare_top_n"), 17)


class TestOpenAICompat(unittest.TestCase):
    def test_extract_text_and_meta_handles_nested_text_value(self) -> None:
        hf = sp.OpenAICompatModel(
            model="gpt-5.4",
            api_base_url="https://api.openai.com/v1",
            api_key="test",
            api_reasoning_effort="auto",
        )
        text, meta = hf._extract_text_and_meta(
            {
                "id": "resp_123",
                "model": "gpt-5.4",
                "usage": {"completion_tokens": 12, "completion_tokens_details": {"reasoning_tokens": 5}},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": [
                                {"type": "output_text", "text": {"value": "hello world"}},
                            ]
                        },
                    }
                ],
            }
        )
        self.assertEqual(text, "hello world")
        self.assertEqual(meta.get("finish_reason"), "stop")
        self.assertEqual(meta.get("completion_tokens"), 12)
        self.assertEqual(meta.get("reasoning_tokens"), 5)
        self.assertEqual(meta.get("empty_output"), False)

    def test_chat_retries_with_openai_gpt5_compat_fallbacks(self) -> None:
        hf = sp.OpenAICompatModel(
            model="gpt-5",
            api_base_url="https://api.openai.com/v1",
            api_key="test",
            api_reasoning_effort="auto",
        )
        seen_payloads: list[dict] = []

        def _fake_post(_url, *, headers, payload, timeout, max_retries):
            seen_payloads.append(dict(payload))
            if len(seen_payloads) == 1:
                raise RuntimeError("HTTP 400: Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.")
            if len(seen_payloads) == 2:
                raise RuntimeError("HTTP 400: Unsupported value: 'temperature' does not support 0.7 with this model. Only the default (1) value is supported.")
            if len(seen_payloads) == 3:
                raise RuntimeError("HTTP 400: Unsupported parameter: 'top_p' is not supported with this model.")
            return {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}

        with patch.object(sp, "_http_post_json", side_effect=_fake_post):
            text = hf.generate_text("hi", max_new_tokens=32, temperature=0.7, top_p=0.95)

        self.assertEqual(text, "ok")
        self.assertEqual(seen_payloads[0].get("max_tokens"), 32)
        self.assertEqual(seen_payloads[1].get("max_completion_tokens"), 32)
        self.assertEqual(seen_payloads[2].get("temperature"), 1.0)
        self.assertNotIn("top_p", seen_payloads[3])

    def test_print_config_only_includes_api_reasoning_effort(self) -> None:
        out, _err = _run_main(
            [
                "--backend",
                "openai_compat",
                "--model",
                "dummy",
                "--query",
                "q",
                "--api_reasoning_effort",
                "none",
                "--print_config_only",
            ]
        )
        obj = json.loads(out)
        cfg = obj.get("cfg") or {}
        self.assertEqual(cfg.get("api_reasoning_effort"), "none")

    def test_generate_api_text_with_reasoning_retry_retries_once(self) -> None:
        class _FakeHF:
            def __init__(self) -> None:
                self.model = "gpt-5.4"
                self.calls: list[int] = []
                self.last_response_meta: dict = {}

            def generate_text(self, prompt: str, *, max_new_tokens: int, **kwargs) -> str:
                _ = (prompt, kwargs)
                self.calls.append(int(max_new_tokens))
                if len(self.calls) == 1:
                    self.last_response_meta = {
                        "finish_reason": "length",
                        "completion_tokens": 160,
                        "reasoning_tokens": 160,
                        "empty_output": True,
                    }
                    return ""
                self.last_response_meta = {
                    "finish_reason": "stop",
                    "completion_tokens": 220,
                    "reasoning_tokens": 120,
                    "empty_output": False,
                }
                return "visible text"

        hf = _FakeHF()
        warnings: list[str] = []

        text, meta = sp._generate_api_text_with_reasoning_retry(
            hf,  # type: ignore[arg-type]
            provider="openai",
            phase="ponder_stage",
            prompt="hello",
            max_new_tokens=160,
            temperature=0.7,
            top_p=0.95,
            top_k=0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            seed=123,
            api_warnings=warnings,
        )

        self.assertEqual(text, "visible text")
        self.assertEqual(hf.calls, [160, 384])
        self.assertEqual(meta.get("auto_retry", {}).get("to_max_tokens"), 384)
        self.assertEqual(len(warnings), 1)

    def test_format_api_http_error_includes_retry_after(self) -> None:
        msg = sp.format_api_http_error(
            sp.APIHTTPError(status=429, body="rate limit exceeded", retry_after_s=3.5),
            provider="mistral",
            model="mistral-large-latest",
            phase="baseline",
        )
        self.assertIn("rate limit", msg.lower())
        self.assertIn("Retry-After=3.5s", msg)


class TestTerminalUX(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["OPENAI_API_KEY"] = "test"
        os.environ["MISTRAL_API_KEY"] = "test"

    def test_provider_preset_resolves_openai_config(self) -> None:
        out, _err = _run_main(
            [
                "--provider",
                "openai",
                "--model",
                "gpt-5.4",
                "--query",
                "q",
                "--print_config_only",
            ]
        )
        obj = json.loads(out)
        cfg = obj.get("cfg") or {}
        self.assertEqual(cfg.get("provider"), "openai")
        self.assertEqual(cfg.get("backend"), "openai_compat")
        self.assertEqual(cfg.get("api_base_url"), "https://api.openai.com/v1")
        self.assertEqual(cfg.get("api_key_env"), "OPENAI_API_KEY")

    def test_provider_openai_gpt5_defaults_ponder_budget(self) -> None:
        out, _err = _run_main(
            [
                "--provider",
                "openai",
                "--model",
                "gpt-5.4",
                "--query",
                "q",
                "--print_config_only",
            ]
        )
        obj = json.loads(out)
        cfg = obj.get("cfg") or {}
        self.assertEqual(cfg.get("ponder_max_new_tokens"), 384)

    def test_main_prints_ponder_logs_and_comparison_summary(self) -> None:
        with _tempdir() as td:
            td_path = Path(td)
            out_path = td_path / "run.json"

            def _baseline(_hf, _cfg, _q, **_kw):
                return "Baseline answer."

            def _ponder(_hf, _cfg, _q, **_kw):
                return (
                    "Pondered answer with a different tradeoff.",
                    [
                        {
                            "band_label": "single",
                            "hop_ix": 0,
                            "pipeline_stage_ix": 0,
                            "ponder_mode": "assoc",
                            "keywords": ["latency budget", "trust"],
                            "ponder_question": "What tension appears between latency and trust?",
                            "ponder_log": "Latency wants velocity. Trust wants a braking distance.",
                        }
                    ],
                    {
                        "memory_selected": [{"query": "earlier"}],
                        "probe_compare": {
                            "js_divergence": 0.12,
                            "overlap_count": 4,
                            "mover_count": 2,
                            "top1_changed": True,
                        },
                    },
                )

            with _Chdir(td_path):
                with patch.object(sp, "run_baseline", side_effect=_baseline), patch.object(
                    sp, "run_ponder_dispatch", side_effect=_ponder
                ):
                    out, _err = _run_main(
                        [
                            "--provider",
                            "openai",
                            "--model",
                            "gpt-5.4",
                            "--query",
                            "q",
                            "--mode",
                            "both",
                            "--json_out",
                            str(out_path),
                        ]
                    )

            self.assertIn("=== PONDER LOGS ===", out)
            self.assertIn("Latency wants velocity. Trust wants a braking distance.", out)
            self.assertIn("=== COMPARISON ===", out)
            self.assertIn("answers_changed=yes", out)
            self.assertNotIn("=== PONDER RECORD(S)", out)

            obj = json.loads(out_path.read_text(encoding="utf-8"))
            comp = obj.get("comparison") or {}
            self.assertEqual(comp.get("answer_changed"), True)
            self.assertEqual(comp.get("memory_selected"), 1)
            self.assertAlmostEqual(comp.get("probe_js_divergence"), 0.12)

    def test_main_reports_rate_limit_without_traceback(self) -> None:
        with patch.object(
            sp,
            "run_baseline",
            side_effect=sp.APIHTTPError(status=429, body="Rate limit exceeded", retry_after_s=2.0),
        ):
            with self.assertRaises(SystemExit) as ctx:
                _run_main(
                    [
                        "--provider",
                        "mistral",
                        "--model",
                        "mistral-large-latest",
                        "--query",
                        "q",
                        "--mode",
                        "baseline",
                    ]
                )
        msg = str(ctx.exception)
        self.assertIn("rate limit", msg.lower())
        self.assertIn("Retry-After=2.0s", msg)


class TestPackBehavior(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["OPENAI_API_KEY"] = "test"

    def test_pack_prefers_json_out_over_config_pack_out_default(self) -> None:
        with _tempdir() as td:
            td_path = Path(td)
            cfg_path = td_path / "cfg.json"
            cfg_path.write_text(json.dumps({"pack_out": "default_pack.json"}), encoding="utf-8")

            out_path = td_path / "pack.json"

            def _baseline(_hf, _cfg, _q, **_kw):
                return "BASE"

            def _ponder(_hf, _cfg, _q, **_kw):
                return ("PONDER", [{"a": 1}, {"b": 2}], {})

            with _Chdir(td_path):
                with patch.object(sp, "run_baseline", side_effect=_baseline), patch.object(
                    sp, "run_ponder_dispatch", side_effect=_ponder
                ):
                    _run_main(
                        [
                            "--backend",
                            "openai_compat",
                            "--model",
                            "dummy",
                            "--query",
                            "q",
                            "--pack",
                            "controls",
                            "--config",
                            str(cfg_path),
                            "--json_out",
                            str(out_path),
                            "--memory",
                            "mem.jsonl",
                        ]
                    )

                self.assertTrue(out_path.exists())
                self.assertFalse((td_path / "default_pack.json").exists())
                obj = json.loads(out_path.read_text(encoding="utf-8"))
                self.assertEqual(obj.get("kind"), "pack")
                self.assertIn("items", obj)

    def test_pack_out_dir_prefers_auto_json_out_over_config_pack_out_default(self) -> None:
        with _tempdir() as td:
            td_path = Path(td)
            cfg_path = td_path / "cfg.json"
            cfg_path.write_text(json.dumps({"pack_out": "default_pack.json"}), encoding="utf-8")

            out_dir = td_path / "artifacts"

            def _baseline(_hf, _cfg, _q, **_kw):
                return "BASE"

            def _ponder(_hf, _cfg, _q, **_kw):
                return ("PONDER", [{"a": 1}], {})

            with _Chdir(td_path):
                with patch.object(sp, "run_baseline", side_effect=_baseline), patch.object(
                    sp, "run_ponder_dispatch", side_effect=_ponder
                ):
                    _run_main(
                        [
                            "--backend",
                            "openai_compat",
                            "--model",
                            "dummy",
                            "--query",
                            "q",
                            "--pack",
                            "controls",
                            "--config",
                            str(cfg_path),
                            "--out_dir",
                            str(out_dir),
                            "--memory",
                            "mem.jsonl",
                        ]
                    )

                self.assertFalse((td_path / "default_pack.json").exists())
                json_files = list(out_dir.glob("*.json"))
                self.assertEqual(len(json_files), 1)
                obj = json.loads(json_files[0].read_text(encoding="utf-8"))
                self.assertEqual(obj.get("kind"), "pack")

                # out_dir auto-names a trace file too.
                trace_files = list(out_dir.glob("*.trace.jsonl"))
                self.assertEqual(len(trace_files), 1)
                trace_txt = trace_files[0].read_text(encoding="utf-8")
                self.assertTrue(trace_txt)
                self.assertIn('"event": "session_end"', trace_txt)

                # out_dir auto-names a trace report too.
                html_files = list(out_dir.glob("*.trace.html"))
                self.assertEqual(len(html_files), 1)
                html_txt = html_files[0].read_text(encoding="utf-8").lower()
                self.assertIn("<html", html_txt)

    def test_pack_resume_skips_completed_items(self) -> None:
        with _tempdir() as td:
            td_path = Path(td)
            out_path = td_path / "pack.json"

            def _baseline(_hf, _cfg, _q, **_kw):
                return "BASE"

            def _ponder(_hf, _cfg, _q, **_kw):
                return ("PONDER", [{"a": 1}], {})

            with _Chdir(td_path):
                with patch.object(sp, "run_baseline", side_effect=_baseline), patch.object(
                    sp, "run_ponder_dispatch", side_effect=_ponder
                ):
                    _run_main(
                        [
                            "--backend",
                            "openai_compat",
                            "--model",
                            "dummy",
                            "--query",
                            "q",
                            "--pack",
                            "controls",
                            "--pack_out",
                            str(out_path),
                            "--memory",
                            "mem.jsonl",
                        ]
                    )

                self.assertTrue(out_path.exists())

                def _should_not_run(*_a, **_kw):
                    raise AssertionError("should have been skipped by --pack_resume")

                with patch.object(sp, "run_baseline", side_effect=_should_not_run), patch.object(
                    sp, "run_ponder_dispatch", side_effect=_should_not_run
                ):
                    _run_main(
                        [
                            "--backend",
                            "openai_compat",
                            "--model",
                            "dummy",
                            "--query",
                            "q",
                            "--pack",
                            "controls",
                            "--pack_out",
                            str(out_path),
                            "--pack_resume",
                            "--memory",
                            "mem.jsonl",
                        ]
                    )


class TestTraceReportProbeCompare(unittest.TestCase):
    def test_trace_report_summarizes_probe_compare(self) -> None:
        with _tempdir() as td:
            td_path = Path(td)
            trace_path = td_path / "trace.jsonl"
            rows = [
                {"ts": "2026-03-07T00:00:00Z", "session_id": "s1", "event": "session_start"},
                {
                    "ts": "2026-03-07T00:00:01Z",
                    "session_id": "s1",
                    "event": "probe_compare_stage",
                    "band_label": "mid",
                    "hop_ix": 0,
                    "stage_ix": 0,
                    "ponder_mode": "counterexample",
                    "memory_chars": 120,
                    "prompt_chars": 240,
                    "top_n": 32,
                    "js_divergence": 0.21,
                    "prev_js_divergence": 0.09,
                    "mover_count": 3,
                    "top1_before": {"token": "alpha", "token_id": 1},
                    "top1_after": {"token": "beta", "token_id": 2},
                },
                {
                    "ts": "2026-03-07T00:00:02Z",
                    "session_id": "s1",
                    "event": "probe_compare",
                    "top_n": 32,
                    "js_divergence": 0.42,
                    "js_divergence_mode": "full_vocab",
                    "overlap_count": 11,
                    "jaccard": 0.55,
                    "mover_count": 4,
                    "entered_count": 2,
                    "exited_count": 1,
                    "top1_before": {"token": "alpha", "token_id": 1},
                    "top1_after": {"token": "gamma", "token_id": 3},
                },
                {"ts": "2026-03-07T00:00:03Z", "session_id": "s1", "event": "session_end"},
            ]
            trace_path.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")

            report = tr.analyze_trace(trace_path, max_records=0, session_id="")
            sessions = report.get("sessions") or []
            self.assertEqual(len(sessions), 1)
            probe = sessions[0].get("probe_compare") or {}
            self.assertEqual(probe.get("stage_count"), 1)
            self.assertEqual(probe.get("final", {}).get("mover_count"), 4)
            self.assertAlmostEqual(probe.get("stage_max_js"), 0.21)

            html = tr.render_html(report)
            self.assertIn("probe compare", html.lower())
            self.assertIn("counterexample", html)
            self.assertIn("0.420000", html)


if __name__ == "__main__":
    unittest.main()
