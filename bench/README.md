# HDFS_v1 Anomaly Detection Benchmark

Benchmarks how well a DeciGrep decision model detects **anomalous log blocks**
in the public **HDFS_v1** dataset from
[LogHub](https://github.com/logpai/loghub).

The dataset is big (~1.4 GB) and is **not stored in git** — you download it
once, as described below.

---

## 1. Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) v0.35.0+ running locally (`http://localhost:11434`)
- The decision model you want to benchmark, pulled first, e.g.:

  ```shell
  ollama pull nimble
  ```

- DeciGrep installed in editable mode (so the `bench` package is importable):

  ```shell
  pip install -e .[dev]
  ```

### Start the Ollama server

The benchmark talks to Ollama over HTTP, so the server must be running before
you start. On Windows/macOS the desktop app starts it automatically; on Linux
either rely on the `systemd` service installed by the Ollama installer
(`systemctl status ollama`) or start it manually:

```shell
ollama serve
```

To benchmark against a remote GPU box instead of localhost, start the server
there with `OLLAMA_HOST=0.0.0.0:11434 ollama serve` and pass its address to
the benchmark with `-u http://<host>:11434`. Note: the first request can take
a few seconds if Ollama or the model has not been started/loaded yet; the
wall-clock time reported by the benchmark includes this warm-up.

## 2. Download the HDFS_v1 dataset

The dataset is **not** part of this repository. Download it from LogHub:

1. Open <https://github.com/logpai/loghub>
2. Enter the **HDFS** folder (HDFS_v1)
3. Follow the *Data Download* link in that folder's README to obtain the
   archive containing `HDFS.log` and `anomaly_label.csv` For example:
```bash
wget -O HDFS_v1.zip "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1"
```

4. Extract the archive into `bench/data/` (the default dataset directory)

The archive may extract with either layout (or even nested one level
deeper); the benchmark **locates both files automatically** by searching the
dataset directory recursively, so you don't need to reorganize anything:

```
bench/data/HDFS_v1/            or:            bench/data/
├── HDFS.log                                  ├── HDFS.log
└── anomaly_label.csv                         └── anomaly_label.csv
```

Any extra preprocessed artifacts in the archive (`HDFS_templates.csv`,
`Event_traces.csv`, `Event_occurrence_matrix.csv`, `HDFS.npz`, …) are simply
ignored.

> 💡 **Keep it out of git.** Add `bench/data/` to your `.gitignore` (it is
> already listed in the repository-level ignore rules for the assistant).

You can keep the files anywhere and point the benchmark at them with
`--dataset DIR` (or the `HDFS_V1_DIR` environment variable).

## 3. Run the benchmark

```shell
python -m bench.hdfs_anomaly --model nimble
```

Or use the installed console command (same thing):

```shell
decigrep-benchmark --model nimble
```

The result is printed to **stdout** and is **auto-saved** into the
`bench/results/` directory with a timestamped filename, e.g.
`bench/results/2026-10-10_0743_nimble.txt` (the final stdout line shows the
exact path; runs in the same minute get a `-1`, `-2`, … suffix). The saved
report includes:

- benchmark name, run date, model version (parameter size, quantization,
  digest from Ollama's `/api/show`)
- hardware specification: CPU brand and GPU(s) (via `nvidia-smi` / `wmic`)
- total wall-clock time of the benchmark execution
- **accuracy, precision, recall, F1-score** of anomaly detection
  (anomaly = positive class) plus the confusion matrix
- the required LogHub citation

Progress messages go to stderr so they never pollute the result.

### Options

| Option | Default | Description |
| --- | --- | --- |
| `-m, --model NAME` | *(required)* | Ollama decision model to benchmark, e.g. `nimble` |
| `-d, --dataset DIR` | `bench/data` | Directory that holds (or contains a subfolder with) `HDFS.log` and `anomaly_label.csv`; searched recursively (env: `HDFS_V1_DIR`) |
| `-u, --url URL` | `http://localhost:11434` | Ollama base URL |
| `-t, --threshold P` | `0.5` | Predict anomaly when `P(yes) > P` |
| `-b, --blocks N` | `200` | Number of blocks to evaluate (stratified half/half) |
| `--no-balance` | off | Sample uniformly instead of balancing classes |
| `--seed N` | `42` | Sampling seed (reproducible runs) |
| `-w, --workers N` | `4` | Concurrent Ollama requests |
| `-r, --retries N` | `2` | Retries per block after a failed request |
| `--timeout S` | `120` | Per-request timeout in seconds |
| `--keep-alive VALUE` | `-1` | Ollama `keep_alive`; `-1` keeps the model loaded |
| `--max-lines-per-block N` | `30` | Log lines per block sent to the model |
| `--instructions TEXT` | default | Custom question wording for the model |

### Examples

```shell
# Small first run (200 balanced blocks, 4 workers)
python -m bench.hdfs_anomaly --model nimble

# More blocks for stabler metrics; keep the model hot between requests
python -m bench.hdfs_anomaly --model tev1 --blocks 1000 --workers 8 --keep-alive -1

# Dataset kept somewhere else
python -m bench.hdfs_anomaly --model nimble --dataset C:/data/HDFS_v1

# Reproduce an earlier run exactly
python -m bench.hdfs_anomaly --model nimble --seed 42 --blocks 200
```

### Example output

```
================================================================================
🔥 Benchmark: HDFS_v1 Anomaly Detection
================================================================================
Date:              2026-10-04T12:30:07+03:00
Model:             nimble
Model version:     nimble, 0.6B params, Q4_K_M, digest=9b3c8f1e2d4a
DeciGrep version:  0.1.0
Dataset:           C:/Users/alexa/Documents/projects/DeciGrep/bench/data/HDFS_v1
Blocks evaluated:  200 (100 anomalous / 100 normal, seed=42)
Lines per block:   up to 30
--------------------------------------------------------------------------------
Hardware:
  CPU:             AMD Ryzen 9 7950X 16-Core Processor, 32 logical CPU(s)
  GPU:             NVIDIA GeForce RTX 4090, 24564MiB
--------------------------------------------------------------------------------
Total time:        6m 12s (372.4s)
--------------------------------------------------------------------------------
Results (anomaly = positive class):
  Accuracy:        0.9250
  Precision:       0.9286
  Recall:          0.9100
  F1-score:        0.9192
  Confusion:       TP=182 FP=14 FN=18 TN=186
--------------------------------------------------------------------------------
🔥 Citation
Please cite the following two papers if you use the loghub datasets in your research.

Loghub: Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu. Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics. IEEE International Symposium on Software Reliability Engineering (ISSRE), 2023.

Loghub-2.0: Zhihan Jiang, Jinyang Liu, Junjie Huang, Yichen Li, Yintong Huo, Jiazhen Gu, Zhuangbin Chen, Jieming Zhu, Michael R. Lyu. A Large-scale Evaluation for Log Parsing Techniques: How Far are We?. ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA), 2024.
================================================================================
```

## How it works

1. `anomaly_label.csv` is loaded into `{block id → is anomaly}` ground truth.
2. `HDFS.log` is streamed once; every line mentioning a `blk_<id>` is attached
   to its block (only labeled blocks are kept, at most `--max-lines-per-block`
   lines each, so memory stays bounded).
3. `--blocks` blocks are sampled — balanced 50/50 anomalous/normal by default
   so precision/recall/F1 stay meaningful even though only ~2.9% of blocks are
   anomalous in the full dataset.
4. Each block is sent as one System One `choice` question
   (*"Does this HDFS log excerpt indicate an anomaly?"*, criteria `yes`/`no`).
5. Predictions are compared with the labels and accuracy, precision, recall
   and F1 are computed (anomaly = positive class). Blocks whose API request
   ultimately failed are excluded and reported separately.

## Cost / runtime hints

- Every block costs exactly one model request, so `--blocks 500` = 500 calls.
- Raising `--workers` uses the GPU more efficiently; results still come out in
  file order.
- `--keep-alive -1` keeps the model loaded between requests and avoids reload
  overhead on multi-run sessions.
- Use the small [`HDFS_2k.log`](https://github.com/logpai/loghub/blob/master/HDFS/HDFS_2k.log)
  plus a matching label file for a quick smoke test of the pipeline (no full
  download needed) — but the official benchmark numbers should come from the
  complete HDFS_v1 files.

## Tests

Pure helper logic (parsing, sampling, metrics, truncation) is covered by unit
tests that need neither the dataset nor Ollama:

```shell
pytest tests/test_benchmark.py