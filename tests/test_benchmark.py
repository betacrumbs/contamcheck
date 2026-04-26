"""Tests for benchmark loading and the Sample abstraction.

These tests hit the real HuggingFace datasets API, so they require an
internet connection. They're intentionally not mocked because the failure
modes we care about (dataset format changes, missing fields) only show up
against the real data.
"""

from __future__ import annotations

from datetime import date

import pytest

from contamcheck.benchmark import Benchmark, Sample, load_benchmark


class TestSample:
    """Sample is a frozen dataclass — these tests exist mainly to enforce
    that property and to document what counts as a valid sample."""

    def test_construction(self) -> None:
        sample = Sample(
            benchmark="gsm8k",
            sample_id="gsm8k-test-0",
            question="What is 2 + 2?",
            answer="4",
        )
        assert sample.benchmark == "gsm8k"
        assert sample.answer == "4"
        assert sample.metadata == {}

    def test_is_frozen(self) -> None:
        """Mutating a Sample must raise. If this test ever fails, downstream
        code may be silently corrupting facts about the benchmark."""
        sample = Sample(
            benchmark="gsm8k",
            sample_id="gsm8k-test-0",
            question="Q",
            answer="A",
        )
        with pytest.raises((AttributeError, Exception)):
            sample.answer = "wrong"  # type: ignore[misc]

    def test_release_date_property(self) -> None:
        sample = Sample(
            benchmark="gsm8k",
            sample_id="x",
            question="Q",
            answer="A",
            metadata={"release_date": date(2021, 10, 27)},
        )
        assert sample.release_date == date(2021, 10, 27)

    def test_release_date_missing(self) -> None:
        sample = Sample(benchmark="gsm8k", sample_id="x", question="Q", answer="A")
        assert sample.release_date is None


class TestLoadBenchmark:
    def test_unknown_benchmark_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown benchmark"):
            load_benchmark("not_a_real_benchmark")

    @pytest.mark.slow
    def test_load_gsm8k_small(self) -> None:
        """Smoke test: load 5 GSM8K samples and verify their structure."""
        bench = load_benchmark("gsm8k", n=5, seed=42)
        assert isinstance(bench, Benchmark)
        assert bench.name == "gsm8k"
        assert len(bench) == 5

        for sample in bench:
            assert sample.benchmark == "gsm8k"
            assert sample.question  # non-empty
            assert sample.answer  # non-empty
            # GSM8K answers are always numeric strings.
            # If this fails, our parsing logic broke.
            assert sample.answer.replace(",", "").replace(".", "").replace("-", "").isdigit(), (
                f"Non-numeric GSM8K answer: {sample.answer!r}"
            )
            assert sample.release_date == date(2021, 10, 27)

    @pytest.mark.slow
    def test_load_gsm8k_seed_reproducibility(self) -> None:
        """Same seed must produce the same sample of questions."""
        bench1 = load_benchmark("gsm8k", n=10, seed=42)
        bench2 = load_benchmark("gsm8k", n=10, seed=42)
        ids1 = [s.sample_id for s in bench1]
        ids2 = [s.sample_id for s in bench2]
        assert ids1 == ids2

    @pytest.mark.slow
    def test_load_gsm8k_different_seeds_differ(self) -> None:
        """Different seeds should (with overwhelming probability) produce
        different samples. If this ever fails, sampling is broken."""
        bench1 = load_benchmark("gsm8k", n=20, seed=42)
        bench2 = load_benchmark("gsm8k", n=20, seed=99)
        ids1 = {s.sample_id for s in bench1}
        ids2 = {s.sample_id for s in bench2}
        assert ids1 != ids2