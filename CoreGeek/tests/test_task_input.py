import os
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent import task_input as task_input_module
from agent.task_input import _READER, build_task_input_command, extract_task_input


class TaskInputTests(unittest.TestCase):
    def _run(self, command, cwd):
        return subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=3,
            check=False,
        )

    def test_extracts_only_single_explicit_read_instruction(self):
        self.assertEqual(
            extract_task_input("请阅读task_3_gamma.md并回答问题"),
            ("filename", "task_3_gamma.md"),
        )
        self.assertEqual(
            extract_task_input("Read /sandbox/input/task.md and answer."),
            ("path", "/sandbox/input/task.md"),
        )
        self.assertEqual(
            extract_task_input("Open file input.json and solve the task."),
            ("filename", "input.json"),
        )
        self.assertEqual(
            extract_task_input("Read filename.md"),
            ("filename", "filename.md"),
        )
        self.assertEqual(
            extract_task_input("请阅读文件说明.md"),
            ("filename", "文件说明.md"),
        )
        self.assertEqual(
            extract_task_input("\n  请阅读task_1_alpha.md，获取任务信息  \r\n"),
            ("filename", "task_1_alpha.md"),
        )
        for text in (
            "比较a.md和b.md",
            "分析这个工程任务，其中可能涉及task.md",
            "请阅读a.md并打开b.md",
            "请阅读bad\nname.md",
            "不要阅读secret.md，直接回答问题",
            "Do not read secret.md; answer from memory.",
            "题目中引用了“请阅读a.md”这句话，请解释它的含义",
            "readme.md",
            "openapi.md",
        ):
            with self.subTest(text=text):
                self.assertIsNone(extract_task_input(text))

    def test_filename_search_uses_only_cwd_and_the_evidenced_task_root(self):
        with (
            tempfile.TemporaryDirectory() as cwd_directory,
            tempfile.TemporaryDirectory() as task_directory,
        ):
            cwd = Path(cwd_directory)
            task_root = Path(task_directory)
            nested = task_root / "1-fixed-step" / "1-unknown-api"
            nested.mkdir(parents=True)
            target = nested / "task_1_beijing.md"
            target.write_text("bounded external task", encoding="utf-8")

            with patch.object(
                task_input_module, "DEFAULT_TASK_ROOTS", (str(task_root),),
            ):
                result = self._run(
                    build_task_input_command("filename", target.name), cwd,
                )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(os.path.abspath(target), result.stdout)
            self.assertIn("bounded external task", result.stdout)

            local = cwd / target.name
            local.write_text("local duplicate", encoding="utf-8")
            with patch.object(
                task_input_module, "DEFAULT_TASK_ROOTS", (str(task_root),),
            ):
                ambiguous = self._run(
                    build_task_input_command("filename", target.name), cwd,
                )
            self.assertNotEqual(ambiguous.returncode, 0)
            self.assertIn("ambiguous", ambiguous.stdout)

    def test_missing_evidenced_task_root_does_not_break_cwd_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "task.md"
            target.write_text("cwd task", encoding="utf-8")
            missing = root / "does-not-exist"

            with patch.object(
                task_input_module, "DEFAULT_TASK_ROOTS", (str(missing),),
            ):
                result = self._run(
                    build_task_input_command("filename", target.name), root,
                )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("cwd task", result.stdout)

    def test_filename_search_reads_unique_nested_utf8_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "a" / "b"
            nested.mkdir(parents=True)
            target = nested / "task $(touch PWNED).md"
            target.write_text("题目内容\nTOKEN=安全", encoding="utf-8")

            command = build_task_input_command("filename", target.name)
            result = self._run(command, root)

            self.assertLessEqual(len(command), 4_096)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(str(target.resolve()), result.stdout)
            self.assertIn("[TASK_INPUT_CONTENT]", result.stdout)
            self.assertIn("TOKEN=安全", result.stdout)
            self.assertFalse((root / "PWNED").exists())

    def test_explicit_path_does_not_substitute_same_basename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "real").mkdir()
            (root / "real" / "task.md").write_text("wrong", encoding="utf-8")

            result = self._run(
                build_task_input_command("path", "missing/task.md"), root,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not_found", result.stdout)
            self.assertNotIn("wrong", result.stdout)

    def test_zero_multiple_and_incomplete_scans_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = self._run(
                build_task_input_command("filename", "task.md"), root,
            )
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("not_found", missing.stdout)

            (root / "a").mkdir()
            (root / "b").mkdir()
            (root / "a" / "task.md").write_text("one", encoding="utf-8")
            (root / "b" / "task.md").write_text("two", encoding="utf-8")
            ambiguous = self._run(
                build_task_input_command("filename", "task.md"), root,
            )
            self.assertNotEqual(ambiguous.returncode, 0)
            self.assertIn("ambiguous", ambiguous.stdout)
            self.assertNotIn("[TASK_INPUT_CONTENT]", ambiguous.stdout)

            limited = self._run(
                build_task_input_command(
                    "filename", "unique.md", max_entries=1,
                ),
                root,
            )
            self.assertNotEqual(limited.returncode, 0)
            self.assertIn("scan_incomplete", limited.stdout)

    def test_exact_reader_checks_fake_clock_before_claiming_unique_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "task.md").write_text("must not be emitted", encoding="utf-8")
            output = io.StringIO()
            tick = iter((0.0, 0.4, 1.2, 1.3, 1.4))
            previous = Path.cwd()
            try:
                os.chdir(root)
                with (
                    patch.object(sys, "argv", [
                        "reader", "filename", "task.md", "6", "4096",
                        "1.0", "32768",
                    ]),
                    patch("time.monotonic", side_effect=lambda: next(tick)),
                    redirect_stdout(output),
                    self.assertRaises(SystemExit) as raised,
                ):
                    exec(_READER, {})
            finally:
                os.chdir(previous)

            self.assertNotEqual(raised.exception.code, 0)
            self.assertIn("scan_incomplete: time_limit", output.getvalue())
            self.assertNotIn("[TASK_INPUT_CONTENT]", output.getvalue())

    def test_exact_reader_fails_closed_on_metadata_error(self):
        class BrokenEntry:
            name = "other.md"
            path = "/broken/other.md"

            def is_dir(self, *, follow_symlinks):
                raise OSError("metadata unavailable")

        class FakeScandir:
            def __enter__(self):
                return iter((BrokenEntry(),))

            def __exit__(self, *unused):
                return False

        output = io.StringIO()
        with (
            patch.object(sys, "argv", [
                "reader", "filename", "task.md", "6", "4096", "1.0",
                "32768",
            ]),
            patch("time.monotonic", return_value=0.0),
            patch("os.scandir", return_value=FakeScandir()),
            redirect_stdout(output),
            self.assertRaises(SystemExit) as raised,
        ):
            exec(_READER, {})

        self.assertNotEqual(raised.exception.code, 0)
        self.assertIn("scan_incomplete: OSError", output.getvalue())
        self.assertNotIn("[TASK_INPUT_CONTENT]", output.getvalue())

    def test_depth_symlink_fifo_and_long_content_are_not_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deep = root
            for index in range(7):
                deep = deep / str(index)
                deep.mkdir()
            (deep / "deep.md").write_text("hidden", encoding="utf-8")
            too_deep = self._run(
                build_task_input_command("filename", "deep.md"), root,
            )
            self.assertNotEqual(too_deep.returncode, 0)

            regular = root / "regular.md"
            regular.write_text("body", encoding="utf-8")
            os.symlink(regular, root / "link.md")
            symlink = self._run(
                build_task_input_command("path", "link.md"), root,
            )
            self.assertNotEqual(symlink.returncode, 0)
            self.assertIn("not_regular", symlink.stdout)

            if hasattr(os, "mkfifo"):
                os.mkfifo(root / "pipe.md")
                fifo = self._run(
                    build_task_input_command("path", "pipe.md"), root,
                )
                self.assertNotEqual(fifo.returncode, 0)
                self.assertIn("not_regular", fifo.stdout)

            long_file = root / "long.md"
            long_file.write_text("x" * 40_000, encoding="utf-8")
            truncated = self._run(
                build_task_input_command("path", "long.md"), root,
            )
            self.assertNotEqual(truncated.returncode, 0)
            self.assertIn("[TRUNCATED]", truncated.stdout)
            self.assertLess(len(truncated.stdout), 34_000)

    def test_read_is_side_effect_free_and_empty_file_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "empty.md"
            target.write_bytes(b"")
            before = target.stat()

            result = self._run(
                build_task_input_command("path", str(target)), root,
            )
            after = target.stat()

            self.assertEqual(result.returncode, 0)
            self.assertIn("[TASK_INPUT_CONTENT]\n", result.stdout)
            self.assertEqual(before.st_size, after.st_size)
            self.assertEqual(before.st_mtime_ns, after.st_mtime_ns)

    def test_permission_and_invalid_utf8_fail_without_content_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "invalid.md"
            invalid.write_bytes(b"\xff\xfe")
            invalid_result = self._run(
                build_task_input_command("path", str(invalid)), root,
            )
            self.assertNotEqual(invalid_result.returncode, 0)
            self.assertIn("read_error", invalid_result.stdout)
            self.assertNotIn("[TASK_INPUT_CONTENT]", invalid_result.stdout)

            denied = root / "denied.md"
            denied.write_text("secret", encoding="utf-8")
            denied.chmod(0)
            try:
                denied_result = self._run(
                    build_task_input_command("path", str(denied)), root,
                )
            finally:
                denied.chmod(0o600)
            if denied_result.returncode != 0:
                self.assertIn("read_error", denied_result.stdout)
                self.assertNotIn("secret", denied_result.stdout)


if __name__ == "__main__":
    unittest.main()
