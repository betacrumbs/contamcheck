"""Benchmark loading and the core Sample abstraction.

A Sample is an immutable record of a benchmark question. Every contamination
signal in the library operates on Sample objects, so this module is the
foundation everything else sits on.

Samples carry a release_date so we can build post-cutoff negative controls:
benchmarks released after a model's training cutoff cannot be contaminated,
which gives us a natural baseline for any audit.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from datasets import load_dataset

from contamcheck.config import SUPPORTED_BENCHMARKS


@dataclass(frozen=True, slots=True)
class Sample:
    """A single benchmark question.

    Frozen because samples are facts about the world, not state to mutate.
    If a downstream function tries to overwrite sample.answer, we want that
    to fail loudly at the assignment, not silently corrupt later results.
    """

    benchmark: str           # e.g. "gsm8k"
    sample_id: str           # stable identifier within the benchmark
    question: str
    answer: str              # the canonical answer string
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def release_date(self) -> date | None:
        """When this benchmark was publicly released. Used for cutoff-based controls."""
        d = self.metadata.get("release_date")
        return d if isinstance(d, date) else None


@dataclass
class Benchmark:
    """A loaded benchmark, exposed as a lazy iterator over Samples.

    Lazy iteration matters because some benchmarks (MATH, HumanEval) are large
    enough that you don't want to materialize all samples in memory if you only
    need 200 of them.
    """

    name: str
    samples: list[Sample]

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


# GSM8K was released by OpenAI in October 2021. Source: arXiv:2110.14168.
# This date is used to determine which models could possibly have seen it
# during training (any model with a cutoff after Oct 2021).
GSM8K_RELEASE_DATE = date(2021, 10, 27)


def _load_gsm8k(split: str, n: int | None, seed: int) -> list[Sample]:
    """Load GSM8K from HuggingFace and convert to Sample objects.

    GSM8K stores answers as multi-line strings ending with "#### <number>".
    We extract just the final numeric answer because that's what evaluators
    check against.
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)

    samples: list[Sample] = []
    for i, row in enumerate(ds):
        # Extract the canonical answer. GSM8K format: "...reasoning...\n#### 42"
        raw_answer = row["answer"]
        if "####" in raw_answer:
            answer = raw_answer.split("####")[-1].strip()
        else:
            # Defensive: should never happen for GSM8K, but if it does we want
            # to know rather than silently produce a malformed Sample.
            raise ValueError(f"GSM8K row {i} missing '####' delimiter: {raw_answer!r}")

        samples.append(
            Sample(
                benchmark="gsm8k",
                sample_id=f"gsm8k-{split}-{i}",
                question=row["question"],
                answer=answer,
                metadata={
                    "release_date": GSM8K_RELEASE_DATE,
                    "raw_answer": raw_answer,  # keep the full reasoning for Signal β later
                    "source": "openai/gsm8k",
                },
            )
        )

    if n is not None and n < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, n)

    return samples


def load_benchmark(
    name: str,
    split: str = "test",
    n: int | None = None,
    seed: int = 42,
) -> Benchmark:
    """Load a benchmark by name.

    Args:
        name: Benchmark identifier, e.g. "gsm8k".
        split: Dataset split, typically "test" or "train".
        n: If provided, randomly sample this many examples (without replacement).
        seed: Seed for the random sample. Required for reproducibility.

    Returns:
        A Benchmark containing Sample objects.

    Raises:
        ValueError: If the benchmark name is not recognized.
    """
    if name not in SUPPORTED_BENCHMARKS:
        supported = sorted(SUPPORTED_BENCHMARKS)
        raise ValueError(
            f"Unknown benchmark: {name!r}. Supported benchmarks: {supported}"
        )

    if name == "gsm8k":
        samples = _load_gsm8k(split=split, n=n, seed=seed)
    else:
        # Should be unreachable given the SUPPORTED_BENCHMARKS check above,
        # but kept for defense-in-depth.
        raise ValueError(f"No loader implemented for benchmark: {name!r}")

    return Benchmark(name=name, samples=samples)