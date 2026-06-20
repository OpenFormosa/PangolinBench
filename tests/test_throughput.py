"""Smoke test for the PangolinBench training-throughput benchmark.

This test runs the real training pipeline and therefore requires UbiTok
('ubitok'). It is skipped automatically when UbiTok is not installed.
"""

from __future__ import annotations

import pytest

ubitok = pytest.importorskip("ubitok")

from pangolinbench.throughput import run_benchmark


def test_run_benchmark_tiny_corpus(tmp_path):
    report = run_benchmark(
        tmp_path,
        shard_count=2,
        lines_per_shard=4,
        worker_count=2,
        merge_count=8,
        min_piece_frequency=1,
        max_unique_pieces_for_merge=8,
        unique_piece_limit_policy="drop_lowest_frequency",
        run_evaluation=False,
    )
    assert report["ok"] is True
