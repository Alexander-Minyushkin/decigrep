"""Unit tests for DeciGrep's pure logic (no network calls required).

The network-facing layer is exercised through mocked clients so the suite
runs without Ollama.
"""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stderr
from unittest.mock import Mock, patch

from decigrep import __version__
from decigrep.client import SystemOneError
from decigrep.matcher import (
    DEFAULT_CRITERIA,
    Criterion,
    build_question,
    decide_match,
    parse_criteria,
    scan_lines,
    skip_reason,
)


def _fake_response(yes: float, no: float) -> dict:
    return {
        "model": "nimble",
        "answers": {
            "match": {
                "type": "choice",
                "choice": "yes" if yes >= no else "no",
                "probabilities": {"yes": yes, "no": no},
                "confidence": max(yes, no),
            }
        },
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }


class ParseCriteriaTests(unittest.TestCase):
    def test_simple_keys(self):
        criteria = parse_criteria("yes,no")
        self.assertEqual([c.key for c in criteria], ["yes", "no"])
        self.assertTrue(all(c.description is None for c in criteria))

    def test_descriptions(self):
        criteria = parse_criteria("yes:It matches,no:It does not")
        self.assertEqual(criteria[0], Criterion("yes", "It matches"))
        self.assertEqual(criteria[1], Criterion("no", "It does not"))

    def test_mixed_and_whitespace(self):
        criteria = parse_criteria(" yes , no:Not really ")
        self.assertEqual(criteria[0], Criterion("yes", None))
        self.assertEqual(criteria[1], Criterion("no", "Not really"))

    def test_too_few(self):
        with self.assertRaises(ValueError):
            parse_criteria("yes")

    def test_duplicate_keys(self):
        with self.assertRaises(ValueError):
            parse_criteria("yes,yes")

    def test_empty_key(self):
        with self.assertRaises(ValueError):
            parse_criteria(":empty,no")

    def test_default_matches_requirement(self):
        self.assertEqual([c.key for c in DEFAULT_CRITERIA], ["yes", "no"])


class MatchLogicTests(unittest.TestCase):
    def test_above_threshold(self):
        self.assertTrue(decide_match({"yes": 0.95, "no": 0.05}, "yes", 0.9))

    def test_exactly_at_threshold_is_not_a_match(self):
        self.assertFalse(decide_match({"yes": 0.9, "no": 0.1}, "yes", 0.9))

    def test_below_threshold(self):
        self.assertFalse(decide_match({"yes": 0.5, "no": 0.5}, "yes", 0.9))

    def test_missing_positive_key(self):
        self.assertFalse(decide_match({"no": 1.0}, "yes", 0.0))


class QuestionBuildingTests(unittest.TestCase):
    def test_question_shape(self):
        question = build_question("error 500", parse_criteria("yes,no"))
        match = question["match"]
        self.assertEqual(match["type"], "choice")
        self.assertIn("error 500", match["instructions"])
        self.assertEqual(match["criteria"], {"yes": None, "no": None})

    def test_custom_instructions_replace_pattern(self):
        question = build_question(
            "error 500", parse_criteria("yes,no"),
            instructions='Is "{pattern}" the main topic of this line?',
        )
        self.assertIn('Is "error 500" the main topic of this line?', question["match"]["instructions"])

    def test_default_instructions_mention_pattern(self):
        question = build_question("error 500", parse_criteria("yes,no"))
        self.assertIn("error 500", question["match"]["instructions"])


class SkipReasonTests(unittest.TestCase):
    def test_blank_line(self):
        self.assertIsNotNone(skip_reason("\n"))

    def test_normal_line(self):
        self.assertIsNone(skip_reason("hello world\n"))


class ScanLinesTests(unittest.TestCase):
    def test_skips_blank_and_reports_errors(self):
        client = Mock()
        client.decide.side_effect = SystemOneError("boom")
        lines = [(1, "hello\n"), (2, "\n"), (3, "world\n")]
        decisions = list(
            scan_lines(
                client,
                lines,
                model="nimble",
                pattern="x",
                criteria=DEFAULT_CRITERIA,
                positive_key="yes",
                threshold=0.9,
                keep_alive="-1",
                retries=0,
                workers=1,
            )
        )
        self.assertTrue(decisions[0].error)
        self.assertFalse(decisions[0].skipped)
        self.assertTrue(decisions[1].skipped)
        self.assertFalse(decisions[2].match)
        self.assertEqual(client.decide.call_count, 2)

    def test_matches_above_threshold_in_order(self):
        client = Mock()
        client.decide.side_effect = [
            _fake_response(yes=0.99, no=0.01),   # line 1 -> match
            _fake_response(yes=0.30, no=0.70),   # line 2 -> no match
        ]
        decisions = list(
            scan_lines(
                client,
                [(1, "a\n"), (2, "b\n")],
                model="nimble",
                pattern="x",
                criteria=DEFAULT_CRITERIA,
                positive_key="yes",
                threshold=0.9,
                keep_alive="-1",
                retries=0,
                workers=1,
            )
        )
        self.assertEqual([d.match for d in decisions], [True, False])

    def test_multiworker_streams_in_order(self):
        import time as _time

        client = Mock()

        def fake_decide(**kwargs):
            state = kwargs["state"]
            if state == "slow":
                _time.sleep(0.2)
            return _fake_response(yes=0.99, no=0.01)

        client.decide.side_effect = fake_decide
        tags = ["slow", "f1", "f2", "f3", "f4"]
        lines = [(i, f"{tag}\n") for i, tag in enumerate(tags, start=1)]
        decisions = list(
            scan_lines(
                client,
                lines,
                model="nimble",
                pattern="x",
                criteria=DEFAULT_CRITERIA,
                positive_key="yes",
                threshold=0.5,
                keep_alive="-1",
                retries=0,
                workers=4,
            )
        )
        # Results are yielded strictly in input order even though later lines
        # finish before the first one.
        self.assertEqual([d.line_number for d in decisions], [1, 2, 3, 4, 5])
        self.assertEqual([d.match for d in decisions], [True] * len(tags))


class CliTests(unittest.TestCase):
    def test_parser_defaults(self):
        from decigrep.cli import build_parser

        args = build_parser().parse_args(["pattern", "file.txt"])
        self.assertEqual(args.model, "nimble")
        self.assertEqual(args.threshold, 0.5)
        self.assertEqual(args.criteria, "yes,no")
        self.assertEqual(args.url, "http://localhost:11434")
        self.assertEqual(args.file, "file.txt")

    def test_main_missing_file_exit_2(self):
        from decigrep.cli import main

        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["pat", "definitely_missing_file_12345.txt"]), 2)

    def test_main_stdin_no_match_exit_1(self):
        from decigrep.cli import main

        client = Mock()
        client.decide.return_value = _fake_response(yes=0.01, no=0.99)
        stdout = io.StringIO()
        with patch("decigrep.cli.SystemOneClient", return_value=client), \
             patch("sys.stdin", io.StringIO("line one\nline two\n")), \
             patch("sys.stdout", stdout), \
             redirect_stderr(io.StringIO()):
            code = main(["some pattern", "-"])
        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")

    def test_main_stdin_match_exit_0(self):
        from decigrep.cli import main

        client = Mock()
        client.decide.return_value = _fake_response(yes=0.99, no=0.01)
        stdout = io.StringIO()
        with patch("decigrep.cli.SystemOneClient", return_value=client), \
             patch("sys.stdin", io.StringIO("line one\nline two\n")), \
             patch("sys.stdout", stdout), \
             redirect_stderr(io.StringIO()):
            code = main(["some pattern", "-"])
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "line one\nline two\n")

    def test_main_invert_match(self):
        from decigrep.cli import main

        client = Mock()
        client.decide.return_value = _fake_response(yes=0.99, no=0.01)
        stdout = io.StringIO()
        with patch("decigrep.cli.SystemOneClient", return_value=client), \
             patch("sys.stdin", io.StringIO("line one\nline two\n")), \
             patch("sys.stdout", stdout), \
             redirect_stderr(io.StringIO()):
            code = main(["-v", "some pattern", "-"])
        self.assertEqual(code, 1)  # nothing printed -> no match exit code
        self.assertEqual(stdout.getvalue(), "")

    def test_line_numbers(self):
        from decigrep.cli import main

        client = Mock()
        client.decide.return_value = _fake_response(yes=0.99, no=0.01)
        stdout = io.StringIO()
        with patch("decigrep.cli.SystemOneClient", return_value=client), \
             patch("sys.stdin", io.StringIO("line one\nline two\n")), \
             patch("sys.stdout", stdout), \
             redirect_stderr(io.StringIO()):
            code = main(["-n", "some pattern", "-"])
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "1:line one\n2:line two\n")

    def test_version(self):
        from decigrep.cli import main

        with self.assertRaises(SystemExit) as ctx:
            main(["--version"])
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()