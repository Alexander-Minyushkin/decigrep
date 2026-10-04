"""HDFS_v1 anomaly-detection benchmark for DeciGrep decision models.

Evaluates how well an Ollama System One decision model detects **anomalous
log blocks** in the HDFS_v1 dataset from LogHub
(https://github.com/logpai/loghub).

The dataset is large (~1.4 GB) and is *not* stored in git: the user
downloads it once and extracts the archive anywhere below a dataset
directory (default ``bench/data``). Typical layouts::

    bench/data/HDFS_v1/
        HDFS.log           # raw Hadoop HDFS log, ~11.1 M lines
        anomaly_label.csv  # BlockId,Label (Normal|Anomaly)

    bench/data/
        HDFS.log
        anomaly_label.csv

The two files are located automatically (recursively), so any extraction
layout — files at the top level or nested in a subfolder — works. Any extra
preprocessed artifacts in the archive (``HDFS_templates.csv``,
``Event_traces.csv``, …) are ignored.

Each sampled block is converted into a ``choice`` question ("is this log
excerpt anomalous?") answered by the decision model; predictions are
compared against the ground-truth labels and accuracy / precision /
recall / F1 are reported on stdout together with benchmark date, model
version, hardware (CPU/GPU) and total wall-clock time.

Usage::

    python -m bench.hdfs_anomaly --model nimble
    python -m bench.hdfs_anomaly --model tev1 --blocks 1000 --workers 8

Requires a running Ollama instance (default http://localhost:11434) with
the model pulled (``ollama pull nimble``).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import requests

from decigrep import __version__ as DECIGREP_VERSION
from decigrep.client import SystemOneClient
from decigrep.matcher import (
    MAX_STATE_BYTES,
    Criterion,
    decide_match,
    evaluate_line,
    parse_criteria,
)

BENCHMARK_NAME = "HDFS_v1 Anomaly Detection"
DEFAULT_DATASET_DIR = os.environ.get("HDFS_V1_DIR", "bench/data")
DEFAULT_OLLAMA_URL = "http://localhost:11434"

#: Regex matching a Hadoop block id inside a log line (e.g. ``blk_-1608999687919862906``).
BLOCK_ID_RE = re.compile(r"\bblk_(-?\d+)\b")

#: Default question wording; criteria keys stay "yes"/"no".
DEFAULT_INSTRUCTIONS = (
    "Does this HDFS log excerpt indicate an anomaly? "
    "Signs of an anomaly include error messages, exceptions, failed or aborted "
    "operations, unexpected terminations and unusual system behavior. "
    "Answer 'yes' only if the excerpt clearly shows such signs, otherwise 'no'."
)

CITATION = """🔥 Citation
Please cite the following two papers if you use the loghub datasets in your research.

Loghub: Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu. Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics. IEEE International Symposium on Software Reliability Engineering (ISSRE), 2023.

Loghub-2.0: Zhihan Jiang, Jinyang Liu, Junjie Huang, Yichen Li, Yintong Huo, Jiazhen Gu, Zhuangbin Chen, Jieming Zhu, Michael R. Lyu. A Large-scale Evaluation for Log Parsing Techniques: How Far are We?. ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA), 2024."""


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def normalize_block_id(value: str) -> str:
    """Normalize a block id so ``"blk_123"`` and ``"123"`` compare equal.

    ``anomaly_label.csv`` stores ids like ``blk_3585376470833578214`` in some
    versions and bare numbers in others.
    """
    value = value.strip().lower()
    if value.startswith("blk_"):
        value = value[len("blk_"):]
    return value


def find_dataset_files(dataset_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Locate ``HDFS.log`` and ``anomaly_label.csv`` below *dataset_dir*.

    LogHub archives extract into varying layouts, so the directory tree is
    walked and the first occurrence of each file is returned. Returns
    ``(None, None)`` for files that could not be found.
    """
    log_path: Optional[str] = None
    label_path: Optional[str] = None
    for root, _dirs, files in os.walk(dataset_dir):
        lowered = {name.lower(): name for name in files}
        if "hdfs.log" in lowered and log_path is None:
            log_path = os.path.join(root, lowered["hdfs.log"])
        if "anomaly_label.csv" in lowered and label_path is None:
            label_path = os.path.join(root, lowered["anomaly_label.csv"])
        if log_path is not None and label_path is not None:
            break
    return log_path, label_path


def load_labels(label_path: str) -> Dict[str, bool]:
    """Read ``anomaly_label.csv`` into ``{normalized block id: is_anomaly}``."""
    labels: Dict[str, bool] = {}
    with open(label_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        first = True
        for row in reader:
            if not row:
                continue
            if first:
                first = False
                if row[0].strip().lower() == "blockid":
                    continue
            block_id = row[0].strip()
            label = row[1].strip().lower() if len(row) > 1 else ""
            if not block_id or label not in ("normal", "anomaly"):
                continue
            labels[normalize_block_id(block_id)] = label == "anomaly"
    return labels


def collect_block_lines(
    log_path: str,
    labels: Mapping[str, bool],
    max_lines_per_block: int,
) -> Tuple[Dict[str, List[str]], int]:
    """Stream ``HDFS.log`` once and keep the first lines of each labeled block.

    Returns ``({normalized block id: excerpt lines}, matched log lines)``.
    Only blocks that have a ground-truth label are kept; blank lines are
    dropped and at most *max_lines_per_block* non-blank lines per block are
    stored so memory stays bounded for the full ~1.4 GB file.
    """
    blocks: Dict[str, List[str]] = {}
    matched = 0
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = BLOCK_ID_RE.search(line)
            if not match:
                continue
            block_id = normalize_block_id(match.group(1))
            if block_id not in labels:
                continue
            matched += 1
            lines = blocks.get(block_id)
            if lines is None:
                lines = blocks[block_id] = []
            elif len(lines) >= max_lines_per_block:
                continue
            if not line.strip():
                continue
            lines.append(line)
    return blocks, matched


def build_excerpt(lines: Sequence[str], max_bytes: int = MAX_STATE_BYTES) -> str:
    """Join block lines and truncate to *max_bytes* UTF-8 bytes (API limit)."""
    excerpt = "".join(lines)
    encoded = excerpt.encode("utf-8")
    if len(encoded) <= max_bytes:
        return excerpt
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def sample_blocks(
    block_ids: Sequence[str],
    labels: Mapping[str, bool],
    *,
    count: int,
    balanced: bool,
    seed: int,
) -> Tuple[List[str], int, int]:
    """Pick up to *count* blocks; returns ``(blocks, n_anomalous, n_normal)``.

    With ``balanced=True`` (default) half of the sample comes from each class
    so precision/recall/F1 remain meaningful on small budgets even though
    anomalies are only ~2.9% of HDFS_v1 blocks. With ``balanced=False`` the
    sample mirrors the dataset's natural class distribution.
    """
    rng = random.Random(seed)
    anomalous = [bid for bid in block_ids if labels[bid]]
    normal = [bid for bid in block_ids if not labels[bid]]
    if balanced:
        half = count // 2
        chosen = rng.sample(anomalous, min(half, len(anomalous)))
        chosen += rng.sample(normal, min(half, len(normal)))
        remaining = count - len(chosen)
        if remaining > 0:
            picked = set(chosen)
            pool = [bid for bid in block_ids if bid not in picked]
            chosen += rng.sample(pool, min(remaining, len(pool)))
        rng.shuffle(chosen)
    else:
        chosen = rng.sample(block_ids, min(count, len(block_ids)))
    n_anomalous = sum(1 for bid in chosen if labels[bid])
    return chosen, n_anomalous, len(chosen) - n_anomalous


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    block_id: str
    predicted_anomaly: bool
    true_anomaly: bool
    error: Optional[str] = None


def classify_block(
    client: SystemOneClient,
    *,
    model: str,
    block_id: str,
    excerpt: str,
    true_anomaly: bool,
    criteria: Sequence[Criterion],
    instructions: str,
    positive_key: str,
    threshold: float,
    keep_alive: Optional[str],
    retries: int,
    retry_delay: float,
) -> Prediction:
    """Ask the model whether one log block is anomalous."""
    probabilities, error = evaluate_line(
        client,
        model=model,
        pattern="",
        line=excerpt,
        criteria=criteria,
        instructions=instructions,
        keep_alive=keep_alive,
        retries=retries,
        retry_delay=retry_delay,
    )
    if error is not None:
        return Prediction(block_id, False, true_anomaly, error)
    return Prediction(
        block_id, decide_match(probabilities, positive_key, threshold), true_anomaly
    )


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int
    evaluated: int
    errors: int


def compute_metrics(predictions: Sequence[Prediction]) -> Metrics:
    """Compute accuracy/precision/recall/F1 with *anomaly* as positive class.

    Predictions that failed at the API level are excluded from the metrics
    and counted separately in ``errors``.
    """
    tp = fp = tn = fn = errors = 0
    for prediction in predictions:
        if prediction.error is not None:
            errors += 1
            continue
        if prediction.predicted_anomaly and prediction.true_anomaly:
            tp += 1
        elif prediction.predicted_anomaly and not prediction.true_anomaly:
            fp += 1
        elif not prediction.predicted_anomaly and prediction.true_anomaly:
            fn += 1
        else:
            tn += 1
    evaluated = tp + fp + tn + fn
    accuracy = (tp + tn) / evaluated if evaluated else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Metrics(accuracy, precision, recall, f1, tp, fp, tn, fn, evaluated, errors)


# ---------------------------------------------------------------------------
# Environment / model metadata
# ---------------------------------------------------------------------------


def _run(cmd: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, errors="replace"
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (completed.stdout or "").strip()


def _linux_cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def detect_cpu() -> str:
    """Best-effort CPU description: brand string plus logical CPU count."""
    name = platform.processor()
    if not name:
        wmic = _run(["wmic", "cpu", "get", "Name", "/value"])
        if wmic:
            name = next(
                (ln.split("=", 1)[1].strip() for ln in wmic.splitlines() if "=" in ln),
                "",
            )
    if not name:
        name = _linux_cpu_name()
    if not name:
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    parts = [name] if name else [platform.machine()]
    cores = os.cpu_count() or 0
    parts.append(f"{cores} logical CPU(s)")
    return ", ".join(parts)


def detect_gpu() -> str:
    """Best-effort GPU description via nvidia-smi / wmic / system_profiler."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        out = _run([nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"])
        gpus = [g.strip() for g in out.splitlines() if g.strip()]
        if gpus:
            return "; ".join(gpus)
    wmic = _run(["wmic", "path", "win32_VideoController", "get", "name"])
    names = [ln.strip() for ln in wmic.splitlines() if ln.strip()]
    if names:
        return ", ".join(names)
    mac = _run(["system_profiler", "SPDisplaysDataType"])
    if mac:
        chips = [
            ln.split(":", 1)[1].strip()
            for ln in mac.splitlines()
            if "Chipset Model" in ln and ":" in ln
        ]
        if chips:
            return ", ".join(chips)
    return "not detected (no nvidia-smi / system GPU query available)"


def get_model_info(base_url: str, model: str) -> str:
    """Fetch the installed model version/digest via ``GET /api/show``."""
    try:
        response = requests.post(f"{base_url}/api/show", json={"model": model}, timeout=30.0)
        if response.status_code != 200:
            return model
        data = response.json()
    except (requests.RequestException, ValueError):
        return model
    details = data.get("details") or {}
    parts = [data.get("model") or model]
    for key in ("parameter_size", "quantization_level"):
        value = details.get(key)
        if value:
            parts.append(str(value))
    digest = data.get("digest")
    if digest:
        parts.append(f"digest={digest[:12]}")
    return ", ".join(parts)


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decigrep-benchmark",
        description=(
            f"{BENCHMARK_NAME} benchmark for DeciGrep decision models. "
            "Downloads the HDFS_v1 dataset yourself from "
            "https://github.com/logpai/loghub — it is not stored in git. "
            "Results (date, model version, hardware, timing, accuracy, "
            "precision, recall, F1) are printed to stdout."
        ),
        epilog=CITATION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        metavar="NAME",
        help="Ollama decision model to benchmark (e.g. nimble); pull it first "
             "with `ollama pull <name>`",
    )
    parser.add_argument(
        "-d", "--dataset",
        default=DEFAULT_DATASET_DIR,
        metavar="DIR",
        help="directory holding (or containing a subfolder with) HDFS.log and "
             "anomaly_label.csv; searched recursively so any extraction "
             "layout works (default: %(default)s, override with HDFS_V1_DIR)",
    )
    parser.add_argument(
        "-u", "--url",
        default=DEFAULT_OLLAMA_URL,
        metavar="URL",
        help="Ollama base URL (default: %(default)s)",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        metavar="P",
        help="predict anomaly when P(yes) > P (default: %(default)s)",
    )
    parser.add_argument(
        "-b", "--blocks",
        type=int,
        default=200,
        metavar="N",
        help="number of blocks to evaluate (default: %(default)s)",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="sample blocks uniformly instead of balancing anomalous/normal classes",
    )
    parser.add_argument("--seed", type=int, default=42, metavar="N", help="sampling seed")
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        metavar="N",
        help="concurrent Ollama requests (default: %(default)s)",
    )
    parser.add_argument("-r", "--retries", type=int, default=2, metavar="N", help="retries per block")
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        metavar="S",
        help="per-request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--keep-alive",
        default="-1",
        metavar="VALUE",
        help="Ollama keep_alive; -1 keeps the model loaded between requests",
    )
    parser.add_argument(
        "--max-lines-per-block",
        type=int,
        default=30,
        metavar="N",
        help="log lines per block sent to the model (default: %(default)s)",
    )
    parser.add_argument(
        "--instructions",
        default=DEFAULT_INSTRUCTIONS,
        metavar="TEXT",
        help="custom question instructions for the model",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not 0.0 < args.threshold <= 1.0:
        parser.error("--threshold must be greater than 0 and at most 1")
    if args.blocks < 1:
        parser.error("--blocks must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.retries < 0:
        parser.error("--retries must not be negative")
    if args.max_lines_per_block < 1:
        parser.error("--max-lines-per-block must be at least 1")

    log_path, label_path = find_dataset_files(args.dataset)
    missing = [
        name
        for name, path in (("HDFS.log", log_path), ("anomaly_label.csv", label_path))
        if path is None
    ]
    if missing:
        print("error: missing dataset files under:", file=sys.stderr)
        print(f"  {os.path.abspath(args.dataset)}", file=sys.stderr)
        print(f"  (missing: {', '.join(missing)})", file=sys.stderr)
        print(
            "Download the HDFS_v1 dataset from https://github.com/logpai/loghub "
            "(HDFS folder) and extract the archive into that directory — "
            "HDFS.log and anomaly_label.csv are detected automatically even "
            "inside nested subfolders.",
            file=sys.stderr,
        )
        return 2

    started = time.perf_counter()
    criteria = parse_criteria("yes:the log block is anomalous,no:the log block is normal")
    positive_key = criteria[0].key
    client = SystemOneClient(base_url=args.url, timeout=args.timeout)

    print(f"Loading labels from {label_path} ...", file=sys.stderr)
    labels = load_labels(label_path)
    print(f"  {len(labels)} labeled blocks", file=sys.stderr)

    print(f"Scanning {log_path} ...", file=sys.stderr)
    blocks, matched_lines = collect_block_lines(log_path, labels, args.max_lines_per_block)
    available = [block_id for block_id in blocks]
    print(
        f"  {len(available)} labeled blocks found "
        f"({matched_lines} matching log lines)",
        file=sys.stderr,
    )
    if not available:
        print("error: no labeled blocks found in the log", file=sys.stderr)
        return 2

    sample, n_anomalous, n_normal = sample_blocks(
        available,
        labels,
        count=args.blocks,
        balanced=not args.no_balance,
        seed=args.seed,
    )

    def work(block_id: str) -> Prediction:
        return classify_block(
            client,
            model=args.model,
            block_id=block_id,
            excerpt=build_excerpt(blocks[block_id]),
            true_anomaly=labels[block_id],
            criteria=criteria,
            instructions=args.instructions,
            positive_key=positive_key,
            threshold=args.threshold,
            keep_alive=args.keep_alive,
            retries=args.retries,
            retry_delay=1.0,
        )

    print(
        f"Evaluating {len(sample)} blocks with model {args.model!r} "
        f"({args.workers} worker(s)) ...",
        file=sys.stderr,
    )
    isatty = getattr(sys.stderr, "isatty", lambda: False)()
    predictions_by_id: Dict[str, Prediction] = {}
    done = 0

    def tick() -> None:
        if isatty:
            sys.stderr.write(
                f"\r  {done}/{len(sample)} blocks evaluated ({done / max(1, len(sample)) * 100:.0f}%)"
            )
            sys.stderr.flush()

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(work, block_id): block_id for block_id in sample}
            for future in as_completed(futures):
                block_id = futures[future]
                predictions_by_id[block_id] = future.result()
                done += 1
                tick()
    else:
        for block_id in sample:
            predictions_by_id[block_id] = work(block_id)
            done += 1
            tick()
    if isatty:
        sys.stderr.write("\n")
    predictions = [predictions_by_id[block_id] for block_id in sample]

    metrics = compute_metrics(predictions)
    elapsed = time.perf_counter() - started
    model_info = get_model_info(args.url, args.model)

    separator = "=" * 78
    rule = "-" * 78
    out = sys.stdout
    print(separator, file=out)
    print(f"🔥 Benchmark: {BENCHMARK_NAME}", file=out)
    print(separator, file=out)
    print(
        f"Date:              "
        f"{datetime.datetime.now().astimezone().isoformat(timespec='seconds')}",
        file=out,
    )
    print(f"Model:             {args.model}", file=out)
    print(f"Model version:     {model_info}", file=out)
    print(f"DeciGrep version:  {DECIGREP_VERSION}", file=out)
    print(f"Dataset:           {os.path.dirname(os.path.abspath(log_path))}", file=out)
    print(
        f"Blocks evaluated:  {len(sample)} "
        f"({n_anomalous} anomalous / {n_normal} normal, seed={args.seed})",
        file=out,
    )
    print(f"Lines per block:   up to {args.max_lines_per_block}", file=out)
    print(rule, file=out)
    print("Hardware:", file=out)
    print(f"  CPU:             {detect_cpu()}", file=out)
    print(f"  GPU:             {detect_gpu()}", file=out)
    print(rule, file=out)
    print(f"Total time:        {format_duration(elapsed)} ({elapsed:.1f}s)", file=out)
    print(rule, file=out)
    print("Results (anomaly = positive class):", file=out)
    print(f"  Accuracy:        {metrics.accuracy:.4f}", file=out)
    print(f"  Precision:       {metrics.precision:.4f}", file=out)
    print(f"  Recall:          {metrics.recall:.4f}", file=out)
    print(f"  F1-score:        {metrics.f1:.4f}", file=out)
    print(
        f"  Confusion:       TP={metrics.tp} FP={metrics.fp} "
        f"FN={metrics.fn} TN={metrics.tn}",
        file=out,
    )
    if metrics.errors:
        print(f"  API errors:      {metrics.errors} (excluded from metrics)", file=out)
    print(rule, file=out)
    print(CITATION, file=out)
    print(separator, file=out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())