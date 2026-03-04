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


if __name__ == "__main__":
    unittest.main()
