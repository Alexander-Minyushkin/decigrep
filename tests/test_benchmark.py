"""Unit tests for the HDFS_v1 benchmark helper logic (no Ollama or dataset).

Everything here exercises pure functions from ``bench.hdfs_anomaly`` so the
suite runs offline.
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from bench.hdfs_anomaly import (
    Prediction,
    build_excerpt,
    classify_block,
    collect_block_lines,
    compute_metrics,
    find_dataset_files,
    format_duration,
    load_labels,
    main,
    normalize_block_id,
    sample_blocks,
)

LOG_LINE = "081109 203518 148 2489418647 RECEIVING BLOCK blk_3585376470833578214 src: /10.251.43.220:50010\n"


class NormalizeBlockIdTests(unittest.TestCase):
    def test_with_prefix(self):
        self.assertEqual(normalize_block_id("blk_3585376470833578214"), "3585376470833578214")

    def test_without_prefix(self):
        self.assertEqual(normalize_block_id("3585376470833578214"), "3585376470833578214")

    def test_negative_id(self):
        self.assertEqual(normalize_block_id("blk_-1608999687919862906"), "-1608999687919862906")

    def test_case_insensitive(self):
        self.assertEqual(normalize_block_id("BLK_123"), "123")


class LoadLabelsTests(unittest.TestCase):
    def _write(self, path, rows):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows)

    def test_bom_header_and_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "anomaly_label.csv")
            # utf-8-sig BOM is written by encoding="utf-8-sig"; simulate it:
            with open(path, "wb") as handle:
                handle.write(b"\xef\xbb\xbfBlockId,Label\r\n")
                handle.write(b"blk_123,Anomaly\r\n")
                handle.write(b"456,Normal\r\n")
            labels = load_labels(path)
            self.assertEqual(labels, {"123": True, "456": False})

    def test_ignores_bad_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "anomaly_label.csv")
            self._write(path, [["BlockId", "Label"], ["blk_1", "Normal"], ["blk_2", "weird"], []])
            labels = load_labels(path)
            self.assertEqual(labels, {"1": False})


class FindDatasetFilesTests(unittest.TestCase):
    @staticmethod
    def _write(directory):
        log = os.path.join(directory, "HDFS.log")
        labels = os.path.join(directory, "anomaly_label.csv")
        with open(log, "w", encoding="utf-8") as handle:
            handle.write("081109 203518 148 2489418647 RECEIVING BLOCK blk_1\n")
        with open(labels, "w", newline="", encoding="utf-8") as handle:
            handle.write("BlockId,Label\n")
        return log, labels

    def test_files_at_top_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            log, labels = self._write(tmp)
            self.assertEqual(find_dataset_files(tmp), (log, labels))

    def test_files_in_nested_subfolder(self):
        with tempfile.TemporaryDirectory() as tmp:
            sub = os.path.join(tmp, "HDFS_v1", "nested")
            os.makedirs(sub)
            log, labels = self._write(sub)
            self.assertEqual(find_dataset_files(tmp), (log, labels))

    def test_case_insensitive_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "hdfs.log")
            labels = os.path.join(tmp, "anomaly_label.csv")
            with open(log, "w", encoding="utf-8") as handle:
                handle.write("x\n")
            with open(labels, "w", newline="", encoding="utf-8") as handle:
                handle.write("BlockId,Label\n")
            self.assertEqual(find_dataset_files(tmp), (log, labels))

    def test_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(find_dataset_files(tmp), (None, None))


class CollectBlockLinesTests(unittest.TestCase):
    def test_groups_lines_by_block_and_caps(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "HDFS.log")
            with open(log, "w", encoding="utf-8") as handle:
                handle.write("no block here\n")
                handle.write(LOG_LINE)
                handle.write("081109 203519 149 2489418648 Served block blk_3585376470833578214 to /x\n")
                handle.write("\n")
                handle.write("081109 203520 150 2489418649 ERROR block blk_-1608999687919862906 failed\n")
            labels = {"3585376470833578214": False, "-1608999687919862906": True}
            blocks, matched = collect_block_lines(log, labels, max_lines_per_block=30)
            self.assertEqual(matched, 3)
            self.assertEqual(set(blocks), {"3585376470833578214", "-1608999687919862906"})
            self.assertEqual(len(blocks["3585376470833578214"]), 2)
            self.assertEqual(len(blocks["-1608999687919862906"]), 1)

    def test_unlabeled_blocks_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "HDFS.log")
            with open(log, "w", encoding="utf-8") as handle:
                handle.write(LOG_LINE)
            blocks, matched = collect_block_lines(log, {"999999": False}, max_lines_per_block=10)
            self.assertEqual(blocks, {})
            self.assertEqual(matched, 0)

    def test_max_lines_per_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "HDFS.log")
            with open(log, "w", encoding="utf-8") as handle:
                for _ in range(5):
                    handle.write(LOG_LINE)
            blocks, matched = collect_block_lines(log, {"3585376470833578214": False}, max_lines_per_block=2)
            self.assertEqual(matched, 5)
            self.assertEqual(len(blocks["3585376470833578214"]), 2)


class BuildExcerptTests(unittest.TestCase):
    def test_short_lines_kept_as_is(self):
        self.assertEqual(build_excerpt(["a\n", "b\n"]), "a\nb\n")

    def test_truncation_respects_byte_limit(self):
        lines = ["x" * 200] * 1000
        excerpt = build_excerpt(lines, max_bytes=1024)
        self.assertLessEqual(len(excerpt.encode("utf-8")), 1024)

    def test_unicode_truncation_is_valid_utf8(self):
        lines = ["😀" * 500] * 1000
        excerpt = build_excerpt(lines, max_bytes=512)
        excerpt.encode("utf-8")  # must not raise
        self.assertLessEqual(len(excerpt.encode("utf-8")), 512)


class SampleBlocksTests(unittest.TestCase):
    def setUp(self):
        self.labels = {f"a{i}": True for i in range(20)} | {f"n{i}": False for i in range(80)}

    def test_balanced_split(self):
        chosen, n_an, n_nm = sample_blocks(list(self.labels), self.labels, count=40, balanced=True, seed=1)
        self.assertEqual(len(chosen), 40)
        self.assertEqual(n_an, 20)
        self.assertEqual(n_nm, 20)

    def test_reproducible(self):
        first, _, _ = sample_blocks(list(self.labels), self.labels, count=30, balanced=True, seed=7)
        second, _, _ = sample_blocks(list(self.labels), self.labels, count=30, balanced=True, seed=7)
        self.assertEqual(first, second)

    def test_unbalanced_reflects_distribution(self):
        chosen, n_an, n_nm = sample_blocks(list(self.labels), self.labels, count=50, balanced=False, seed=3)
        self.assertEqual(len(chosen), 50)
        self.assertLess(n_an, n_nm)

    def test_short_pools_are_capped_at_pool_size(self):
        chosen, n_an, n_nm = sample_blocks(["a0", "a1"], self.labels, count=100, balanced=True, seed=1)
        self.assertEqual(len(chosen), 2)
        self.assertEqual(n_an, 2)
        self.assertEqual(n_nm, 0)


class ComputeMetricsTests(unittest.TestCase):
    def _pred(self, predicted, true, error=None):
        return Prediction(block_id="x", predicted_anomaly=predicted, true_anomaly=true, error=error)

    def test_perfect(self):
        predictions = [self._pred(True, True), self._pred(False, False)]
        metrics = compute_metrics(predictions)
        self.assertEqual((metrics.accuracy, metrics.precision, metrics.recall, metrics.f1), (1.0, 1.0, 1.0, 1.0))

    def test_hand_computed(self):
        # TP=2 FP=1 FN=1 TN=2
        predictions = [
            self._pred(True, True),
            self._pred(True, True),
            self._pred(True, False),
            self._pred(False, True),
            self._pred(False, False),
            self._pred(False, False),
        ]
        metrics = compute_metrics(predictions)
        self.assertEqual(metrics.tp, 2)
        self.assertEqual(metrics.fp, 1)
        self.assertEqual(metrics.fn, 1)
        self.assertEqual(metrics.tn, 2)
        self.assertAlmostEqual(metrics.accuracy, 4 / 6)
        self.assertAlmostEqual(metrics.precision, 2 / 3)
        self.assertAlmostEqual(metrics.recall, 2 / 3)
        self.assertAlmostEqual(metrics.f1, 2 / 3)

    def test_errors_excluded(self):
        predictions = [self._pred(True, True), self._pred(False, False, error="boom")]
        metrics = compute_metrics(predictions)
        self.assertEqual(metrics.errors, 1)
        self.assertEqual(metrics.evaluated, 1)
        self.assertEqual((metrics.accuracy, metrics.precision, metrics.recall, metrics.f1), (1.0, 1.0, 1.0, 1.0))

    def test_zero_division_guards(self):
        metrics = compute_metrics([self._pred(False, False)])
        self.assertEqual((metrics.precision, metrics.recall, metrics.f1), (0.0, 0.0, 0.0))
        metrics = compute_metrics([])
        self.assertEqual((metrics.accuracy, metrics.precision, metrics.recall, metrics.f1), (0.0, 0.0, 0.0, 0.0))


class ClassifyBlockTests(unittest.TestCase):
    def test_match_above_threshold(self):
        client = object()
        with patch(
            "bench.hdfs_anomaly.evaluate_line",
            return_value=({"yes": 0.9, "no": 0.1}, None),
        ) as mocked:
            prediction = classify_block(
                client,
                model="nimble",
                block_id="b1",
                excerpt="ERROR\n",
                true_anomaly=True,
                criteria=[],
                instructions="Q",
                positive_key="yes",
                threshold=0.5,
                keep_alive=None,
                retries=0,
                retry_delay=0.0,
            )
        self.assertTrue(prediction.predicted_anomaly)
        self.assertIsNone(prediction.error)
        mocked.assert_called_once()

    def test_error_propagates(self):
        with patch(
            "bench.hdfs_anomaly.evaluate_line",
            return_value=({}, "boom"),
        ):
            prediction = classify_block(
                object(),
                model="nimble",
                block_id="b1",
                excerpt="ERROR\n",
                true_anomaly=True,
                criteria=[],
                instructions="Q",
                positive_key="yes",
                threshold=0.5,
                keep_alive=None,
                retries=0,
                retry_delay=0.0,
            )
        self.assertEqual(prediction.error, "boom")


class FormatDurationTests(unittest.TestCase):
    def test_seconds(self):
        self.assertEqual(format_duration(12.7), "12s")

    def test_minutes(self):
        self.assertEqual(format_duration(372.4), "6m 12s")

    def test_hours(self):
        self.assertEqual(format_duration(3725.0), "1h 02m 05s")


class _FakeOllamaHandler(BaseHTTPRequestHandler):
    """Tiny fake Ollama: /api/show + /v1/systemone, no external services."""

    def log_message(self, *args):  # keep test output quiet
        pass

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw) if raw else {}
        except ValueError:
            data = {}
        if self.path == "/api/show":
            self._send(
                {
                    "model": data.get("model", "nimble"),
                    "digest": "0123456789abcdef",
                    "details": {"parameter_size": "0.6B", "quantization_level": "Q4_K_M"},
                }
            )
            return
        if self.path == "/v1/systemone":
            state = data.get("state", "")
            anomaly = "ERROR" in state
            yes = 0.95 if anomaly else 0.05
            self._send(
                {
                    "answers": {
                        "match": {
                            "type": "choice",
                            "choice": "yes" if anomaly else "no",
                            "probabilities": {"yes": yes, "no": 1 - yes},
                        }
                    }
                }
            )
            return
        self._send({"error": "not found"}, 404)


class MainEndToEndTests(unittest.TestCase):
    def test_main_runs_and_reports_all_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            label_path = os.path.join(tmp, "anomaly_label.csv")
            with open(label_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["BlockId", "Label"])
                writer.writerow(["blk_1", "Anomaly"])  # predicted anomaly  -> TP
                writer.writerow(["blk_2", "Normal"])   # predicted anomaly  -> FP
                writer.writerow(["blk_3", "Anomaly"])  # predicted normal   -> FN
                writer.writerow(["blk_4", "Normal"])   # predicted normal   -> TN
            log_path = os.path.join(tmp, "HDFS.log")
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("081109 203518 148 1 ERROR blk_1 src:/x\n")
                handle.write("081109 203519 149 2 ERROR blk_2 src:/x\n")
                handle.write("081109 203520 150 3 Served block blk_3 to /x\n")
                handle.write("081109 203521 151 4 Served block blk_4 to /x\n")

            server = HTTPServer(("127.0.0.1", 0), _FakeOllamaHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                stdout, stderr = io.StringIO(), io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = main(
                        [
                            "--model", "nimble",
                            "--dataset", tmp,
                            "--url", url,
                            "--blocks", "4",
                            "--seed", "1",
                            "--workers", "2",
                        ]
                    )
                self.assertEqual(code, 0, msg=stderr.getvalue())
                out = stdout.getvalue()
                self.assertIn("🔥 Benchmark: HDFS_v1 Anomaly Detection", out)
                self.assertIn("Model version:", out)
                self.assertIn("digest=0123456789ab", out)
                self.assertIn("CPU:", out)
                self.assertIn("GPU:", out)
                self.assertIn("Total time:", out)
                self.assertIn("Accuracy:", out)
                self.assertIn("Precision:", out)
                self.assertIn("Recall:", out)
                self.assertIn("F1-score:", out)
                self.assertIn("TP=1 FP=1 FN=1 TN=1", out)
                self.assertIn("Loghub:", out)  # required citation present
                self.assertIn("Loghub-2.0:", out)
            finally:
                server.shutdown()
                server.server_close()

    def test_main_fails_gracefully_without_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = main(["--model", "nimble", "--dataset", tmp])
            self.assertEqual(code, 2)
            self.assertIn("missing dataset files", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()