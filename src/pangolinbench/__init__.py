"""PangolinBench tokenizer evaluation suite."""

from .runner import run_pangolinbench
from .schema import BenchmarkExample, CorpusManifest, LexiconItem
from .tokenizer_adapter import ByteRoundtripTestAdapter, ReferenceSegmentationAdapter, TokenizerAdapter

BENCHMARK_NAME = "PangolinBench"
BENCHMARK_VERSION = "0.1.0"

__all__ = [
    "BENCHMARK_NAME",
    "BENCHMARK_VERSION",
    "BenchmarkExample",
    "ByteRoundtripTestAdapter",
    "CorpusManifest",
    "LexiconItem",
    "ReferenceSegmentationAdapter",
    "TokenizerAdapter",
    "run_pangolinbench",
]
