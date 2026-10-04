"""Benchmarks for DeciGrep decision models.

The benchmark suite evaluates how well an Ollama System One decision model
performs anomaly detection on the public **HDFS_v1** dataset from LogHub
(https://github.com/logpai/loghub).

The dataset is large and is **not** stored in git — the user downloads it
once and points the benchmark at it. See ``bench/README.md``.
"""