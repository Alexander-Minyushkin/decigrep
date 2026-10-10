# DeciGrep

A grep-like command-line utility that finds lines in a file matching a
pattern, using **Ollama decision models** (the System One API, e.g.
[`nimble`](https://ollama.com/library/nimble)) instead of regular
expressions.

Each line is sent to the model together with the pattern. The model chooses
between the configured criteria (by default `yes` / `no`), and the line is
printed to standard output when the probability of the *positive* criterion
(`yes` by default) is greater than the confidence threshold (`0.5` by
default).

```
$ decigrep "payment failed" sample.log
Our checkout has returned 500 errors since 9am.
Card payment declined for order #12345.
```

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com/) v0.35.0 or later running locally
  (`http://localhost:11434`)
- A decision model, for example:

  ```shell
  ollama pull nimble
  ```

### Starting Ollama

Ollama must be running before you invoke DeciGrep. On Windows and macOS the
desktop app starts a background server automatically; on Linux (or a headless
server/VM) either let the `systemd` service from the installer handle it or
start the server manually:

```shell
ollama serve
```

By default it listens on `http://localhost:11434`; use `OLLAMA_HOST` (e.g.
`OLLAMA_HOST=0.0.0.0:11434 ollama serve`) to bind another interface. If
DeciGrep starts before the server is up, the first request may take a few
seconds or fail with "could not reach Ollama" — run
`decigrep -V ...` for verbose progress.

## Installation

```shell
pip install .
# or, editable for development:
pip install -e .[dev]
```

After installation the `decigrep` command is available. You can also run it
without installing:

```shell
python -m decigrep --help
```

## Usage

```
decigrep [OPTIONS] PATTERN FILE
```

`PATTERN` is a plain-text pattern, **not** a regular expression — matching is
semantic (the model judges whether the line "means" the pattern). `FILE` is
the file to scan, or `-` to read from standard input.

### Options

| Option | Default | Description |
| --- | --- | --- |
| `-m, --model` | `nimble` | Ollama decision model to use |
| `-u, --url` | `http://localhost:11434` | Ollama base URL |
| `-t, --threshold P` | `0.5` | Print a line when `P(positive) > P` |
| `-c, --criteria SPEC` | `yes,no` | Comma-separated `key[:description]` criteria; the first key is the positive one |
| `--instructions TEXT` | default | Custom question instructions for the model; `{pattern}` is replaced with the search pattern |
| `-n, --line-number` | off | Prefix printed lines with their 1-based line number |
| `-v, --invert-match` | off | Print lines that do **not** match |
| `-w, --workers N` | `1` | Number of concurrent Ollama requests |
| `-r, --retries N` | `2` | Retries per line after a failed request |
| `--timeout S` | `60` | Per-request timeout in seconds |
| `--keep-alive VALUE` | `-1` | Ollama `keep_alive`; `-1` keeps the model loaded between requests |
| `-V, --verbose` | off | Print progress, per-line probabilities and skip reasons to stderr |
| `-q, --quiet` | off | Suppress progress and warnings on stderr |
| `--version` | — | Show version and exit |
| `-h, --help` | — | Show help and exit |

### Exit status

Like `grep`:

| Code | Meaning |
| --- | --- |
| `0` | At least one line was printed |
| `1` | No line matched |
| `2` | Error (missing file, invalid options, failed API calls) |

## Examples

```shell
# Print lines about refunds (semantic matching, not literal text)
decigrep "refund request" tickets.txt

# Show line numbers, as with grep -n
decigrep -n "urgent outage" incidents.log

# Print lines that do NOT match
decigrep -v "routine" tickets.txt

# Use custom criteria (the first key is the positive one)
decigrep -c "relevant:Line is relevant,irrelevant:Line is irrelevant" notes.txt

# Custom threshold and model
decigrep -t 0.8 -m tev1 "database error" app.log

# Custom question wording (the {pattern} placeholder is substituted)
decigrep --instructions 'Is "{pattern}" the main topic of this line?' notes.txt

# Scan faster with concurrent requests (matching lines are printed on the
# go, as soon as they are decided; output order always follows the file)
decigrep -w 4 "payment failed" transactions.log

# Inspect the model's probabilities for every line
decigrep -V "bug report" issues.txt

# Read from stdin
cat log.txt | decigrep "service is down" -
```

## How it works

For every non-blank line, DeciGrep sends one request to
`POST <url>/v1/systemone`:

```json
{
  "model": "nimble",
  "state": "Our checkout has returned 500 errors since 9am.",
  "questions": {
    "match": {
      "type": "choice",
      "instructions": "Does the line match the pattern \"payment failed\"? Judge by meaning and topic, not just exact wording.",
      "criteria": { "yes": null, "no": null }
    }
  }
}
```

The line is printed when `answers.match.probabilities["yes"] > threshold`.

Notes:

- Blank lines are skipped (the API requires a non-empty `state`).
- Lines longer than 64 KiB are skipped (the image-less request limit).
- The full API contract is documented in [`docs/Ollama-SystemOne.md`](docs/Ollama-SystemOne.md)
  and [`docs/Ollama-Decision.md`](docs/Ollama-Decision.md).

## Benchmarks

An anomaly-detection benchmark for the HDFS_v1 dataset from
[LogHub](https://github.com/logpai/loghub) lives in
[`bench/`](bench/README.md). It measures accuracy, precision, recall and
F1-score of a decision model on HDFS log blocks and prints the full report
(date, model version, CPU/GPU, total time, metrics, citation) to stdout:

```shell
# 1. download HDFS_v1 yourself (see bench/README.md) and extract it into bench/data
# 2. run the benchmark against a decision model
python -m bench.hdfs_anomaly --model nimble
# or, after `pip install -e .`, the installed console command:
decigrep-benchmark --model nimble
```

Every run also **auto-saves** its report to the `bench/results/` directory
(e.g. `bench/results/2026-10-10_0743_nimble.txt`) and prints the saved path
at the end.

The ~1.4 GB dataset is **not** stored in git — you download it once from
LogHub and extract it into `bench/data` (overridable via `--dataset` or the
`HDFS_V1_DIR` environment variable). `HDFS.log` and `anomaly_label.csv` are
located automatically even when the archive extracts into a nested
subfolder. Full instructions, CLI options and example output are in
[`bench/README.md`](bench/README.md).

## Development

```shell
pip install -e .[dev]
pytest
```

For instructions on provisioning a GCP Compute Engine VM with an NVIDIA L4
GPU and running the tests and live smoke tests there, see
[`docs/GCP_TESTING.md`](docs/GCP_TESTING.md).

## License

[MIT](LICENSE) — free to use in your own projects.
