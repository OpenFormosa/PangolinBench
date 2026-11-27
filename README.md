# PangolinBench

PangolinBench is a tokenizer benchmark for Taiwan-centric language modeling.
It measures correctness, compression, lexicon fertility, over-segmentation,
script coverage, rich transcription stability, and optional throughput.

The package is `pangolinbench`.

PangolinBench does not claim that byte-level roundtrip alone is enough. A
tokenizer can encode any UTF-8 text yet still be inefficient for Traditional
Chinese, Bopomofo, Taigi romanization, Hakka romanization, or Taiwan-specific
terms. PangolinBench therefore reports both roundtrip correctness and efficiency
diagnostics. PangolinBench also reports tokenizer vocabulary size when available,
because smaller vocabularies reduce embedding/output-head cost even when token
counts are similar.

Built-in PangolinBench fixtures are synthetic, inspectable, deterministic, and
license-clean. Real evaluation should plug in heldout corpora through a manifest
rather than committing private, licensed, or large corpora to this repository.
The benchmark avoids dense timestamp token ranges and validates rich
transcription JSON stability.

Run PangolinBench with a tokenizer:

```bash
python -m pangolinbench.cli \
  --tokenizer path/to/tokenizer \
  --config configs/pangolinbench.yaml \
  --output reports/pangolinbench_report.json
```

Run a local smoke test with the byte roundtrip test adapter:

```bash
python -m pangolinbench.cli \
  --byte-roundtrip-test-adapter \
  --config configs/pangolinbench.yaml \
  --output reports/pangolinbench_report.json
```

The compatibility evaluator also accepts PangolinBench flags:

```bash
python evaluate_tokenizer.py \
  --tokenizer path/to/tokenizer \
  --pangolinbench \
  --pangolinbench-config configs/pangolinbench.yaml \
  --output reports/evaluation_report.json
```

## Training-throughput benchmark

PangolinBench also ships a deterministic synthetic training-throughput
benchmark that measures how fast a tokenizer trainer ingests a synthetic
mixed-script corpus. Unlike the quality benchmark (pure stdlib), the
throughput benchmark trains a real tokenizer and therefore needs UbiTok:

```bash
pip install ubitok
```

Run it with the installed console script:

```bash
pangolinbench-throughput \
  --output-dir reports/throughput \
  --shard-count 8 \
  --lines-per-shard 1000 \
  --merge-count 128
```

`import pangolinbench` and the quality benchmark never import `ubitok`; the
dependency is loaded lazily only when the throughput benchmark actually runs.

## License

Apache License 2.0 © 2026 OpenFormosa contributors. See [LICENSE](LICENSE). Evaluation corpora referenced through manifests keep their own licenses.
