"""Deterministic synthetic training-throughput benchmark for PangolinBench.

This module ports the synthetic tokenizer-training benchmark. The pure-stdlib
helpers (corpus generation, progress summarization, argument parsing) have no
third-party dependencies. The actual training run requires UbiTok ('ubitok'),
which is imported lazily inside ``run_benchmark`` so that ``import pangolinbench``
and the quality benchmark never depend on it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _load_ubitok():
    try:
        from ubitok.config import TokenizerTrainingConfig
        from ubitok.progress import get_resource_usage
        from ubitok.special_tokens import DEFAULT_SPECIAL_TOKENS as SPECIAL_TOKENS
        from ubitok.pipeline import train_tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "PangolinBench training-throughput benchmark requires UbiTok ('ubitok'). "
            "Install with `pip install ubitok` or run only the quality benchmark."
        ) from exc
    return TokenizerTrainingConfig, get_resource_usage, SPECIAL_TOKENS, train_tokenizer


def write_synthetic_corpus(
    output_dir: str | Path,
    *,
    shard_count: int,
    lines_per_shard: int,
) -> list[Path]:
    """Write deterministic mixed-script text shards and return their paths."""
    corpus_dir = Path(output_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for shard_index in range(shard_count):
        path = corpus_dir / f"synthetic_{shard_index:05d}.txt"
        with path.open("w", encoding="utf-8") as handle:
            for line_index in range(lines_per_shard):
                handle.write(_synthetic_line(shard_index, line_index))
                handle.write("\n")
        paths.append(path)
    return paths


def run_benchmark(
    output_dir: str | Path,
    *,
    shard_count: int = 8,
    lines_per_shard: int = 1000,
    worker_count: int = 4,
    count_shard_count: int = 16,
    flush_threshold: int = 10000,
    merge_count: int = 128,
    min_piece_frequency: int = 2,
    max_unique_pieces_for_merge: int | None = None,
    unique_piece_limit_policy: str = "error",
    split_large_files: bool = True,
    split_min_bytes: int = 512 * 1024 * 1024,
    split_chunk_bytes: int = 256 * 1024 * 1024,
    merge_sequence_storage_backend: str = "array",
    merge_pair_key_backend: str = "int",
    run_evaluation: bool = True,
) -> dict[str, Any]:
    """Run synthetic training and write a benchmark_report.json summary."""
    TokenizerTrainingConfig, get_resource_usage, SPECIAL_TOKENS, train_tokenizer = _load_ubitok()
    benchmark_dir = Path(output_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    corpus_paths = write_synthetic_corpus(
        benchmark_dir / "synthetic_corpus",
        shard_count=shard_count,
        lines_per_shard=lines_per_shard,
    )
    tokenizer_dir = benchmark_dir / "tokenizer"
    config = TokenizerTrainingConfig(
        vocab_size=256 + len(SPECIAL_TOKENS) + merge_count,
        min_piece_frequency_for_merge=min_piece_frequency,
        piece_count_backend="sharded",
        piece_count_flush_threshold=flush_threshold,
        piece_count_worker_count=worker_count,
        deterministic_shard_count=count_shard_count,
        merge_learner_backend="incremental",
        pair_position_index_backend="array",
        merge_sequence_storage_backend=merge_sequence_storage_backend,
        merge_pair_key_backend=merge_pair_key_backend,
        merge_heap_rebuild_multiplier=8,
        progress_report_interval_seconds=0.0,
        progress_report_every_merges=max(1, merge_count // 16),
        progress_log_jsonl="benchmark_progress.jsonl",
        exact_mode=False,
        max_unique_pieces_for_merge=max_unique_pieces_for_merge,
        unique_piece_limit_policy=unique_piece_limit_policy,
        piece_count_split_large_files=split_large_files,
        piece_count_split_min_bytes=split_min_bytes,
        piece_count_split_chunk_bytes=split_chunk_bytes,
    )
    started = time.monotonic()
    result = train_tokenizer(
        corpus_paths,
        tokenizer_dir,
        config=config,
        run_evaluation=run_evaluation,
    )
    elapsed = time.monotonic() - started
    progress_path = tokenizer_dir / "benchmark_progress.jsonl"
    report = {
        "ok": True,
        "elapsed_seconds": round(elapsed, 6),
        "corpus": {
            "shard_count": shard_count,
            "lines_per_shard": lines_per_shard,
            "total_lines": shard_count * lines_per_shard,
            "total_bytes": sum(path.stat().st_size for path in corpus_paths),
        },
        "config": {
            "worker_count": worker_count,
            "count_shard_count": count_shard_count,
            "flush_threshold": flush_threshold,
            "merge_count": merge_count,
            "min_piece_frequency": min_piece_frequency,
            "max_unique_pieces_for_merge": max_unique_pieces_for_merge,
            "unique_piece_limit_policy": unique_piece_limit_policy,
            "split_large_files": split_large_files,
            "split_min_bytes": split_min_bytes,
            "split_chunk_bytes": split_chunk_bytes,
            "merge_sequence_storage_backend": merge_sequence_storage_backend,
            "merge_pair_key_backend": merge_pair_key_backend,
        },
        "training": {
            "actual_vocab_size": len(result.vocab),
            "number_of_merges_learned": len(result.merges),
            "output_dir": str(tokenizer_dir),
        },
        "progress": _summarize_progress(progress_path),
        "resource_usage": get_resource_usage(),
    }
    (benchmark_dir / "benchmark_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the synthetic tokenizer training benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--lines-per-shard", type=int, default=1000)
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--count-shard-count", type=int, default=16)
    parser.add_argument("--flush-threshold", type=int, default=10000)
    parser.add_argument("--merge-count", type=int, default=128)
    parser.add_argument("--min-piece-frequency", type=int, default=2)
    parser.add_argument("--max-unique-pieces-for-merge", type=int, default=None)
    parser.add_argument(
        "--unique-piece-limit-policy",
        choices=("error", "drop_lowest_frequency"),
        default="error",
    )
    parser.add_argument(
        "--no-split-large-files",
        action="store_true",
        help="Disable byte-range splitting for large input files.",
    )
    parser.add_argument("--split-min-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--split-chunk-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument(
        "--merge-sequence-storage-backend",
        choices=("array", "list"),
        default="array",
    )
    parser.add_argument(
        "--merge-pair-key-backend",
        choices=("int", "tuple"),
        default="int",
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args(argv)
    report = run_benchmark(
        args.output_dir,
        shard_count=args.shard_count,
        lines_per_shard=args.lines_per_shard,
        worker_count=args.worker_count,
        count_shard_count=args.count_shard_count,
        flush_threshold=args.flush_threshold,
        merge_count=args.merge_count,
        min_piece_frequency=args.min_piece_frequency,
        max_unique_pieces_for_merge=args.max_unique_pieces_for_merge,
        unique_piece_limit_policy=args.unique_piece_limit_policy,
        split_large_files=not args.no_split_large_files,
        split_min_bytes=args.split_min_bytes,
        split_chunk_bytes=args.split_chunk_bytes,
        merge_sequence_storage_backend=args.merge_sequence_storage_backend,
        merge_pair_key_backend=args.merge_pair_key_backend,
        run_evaluation=not args.skip_evaluation,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _synthetic_line(shard_index: int, line_index: int) -> str:
    speaker = (shard_index + line_index) % 4
    start = line_index * 3.5
    end = start + 3.42
    variants = [
        "大家好，歡迎回來。",
        "今天我們要討論台灣本土語音模型。",
        "注音 ㄅㄆㄇ 與台語 tshit-á 混合測試。",
        "JSON punctuation: {\"Start\":0.00,\"End\":3.42}.",
        "[Silence] [Noise] [Music] timestamp 3575.50",
    ]
    content = variants[(shard_index * 17 + line_index) % len(variants)]
    return (
        f'{{"Start":{start:.2f},"End":{end:.2f},'
        f'"Speaker":{speaker},"Content":"{content}"}}'
    )


def _summarize_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"record_count": 0, "phases": [], "peak_rss_bytes": None}
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    peak_values = [
        record["peak_rss_bytes"]
        for record in records
        if record.get("peak_rss_bytes") is not None
    ]
    return {
        "record_count": len(records),
        "phases": sorted({record["phase"] for record in records}),
        "peak_rss_bytes": max(peak_values) if peak_values else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
