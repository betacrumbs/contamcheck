import time
from typing import List, Dict
from dataclasses import dataclass, asdict
from src.benchmark import BenchmarkSample, load_gsm8k
from src.model import query_model, extract_answer
from src.perturbations import generate_all_perturbations


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class SampleResult:
    """Results for a single sample across all perturbation levels."""
    sample_id: int
    source: str
    original_answer: str
    model_answers: Dict[str, str]     # level_name -> model's answer
    expected_answers: Dict[str, str]  # level_name -> ground truth
    correctness: Dict[str, bool]      # level_name -> was model correct


@dataclass
class ContaminationScore:
    """Aggregated contamination metrics for one model × one benchmark."""
    model_name: str
    benchmark: str
    n_samples: int
    accuracy_by_level: Dict[str, float]   # level -> accuracy across samples
    accuracy_gap: Dict[str, float]        # level -> original_acc - level_acc
    contamination_index: float            # average gap, higher = more contaminated


# ============================================================
# CORE EVALUATION
# ============================================================

def evaluate_sample(
    sample: BenchmarkSample,
    sample_id: int,
    model_name: str = "llama-3.1-8b-instant",
    verbose: bool = False
) -> SampleResult:
    """
    Generate all perturbations for one sample, query the model on each,
    and record correctness per level.
    """
    if verbose:
        print(f"\nSample {sample_id} [{sample.source}]")
        print(f"  Q: {sample.question[:80]}...")
        print(f"  Expected: {sample.answer}")

    variants = generate_all_perturbations(sample)

    model_answers = {}
    expected_answers = {}
    correctness = {}

    for level_name, variant in variants.items():
        # query model with retry on rate limit
        raw_response = _query_with_retry(variant, model_name)
        extracted = extract_answer(raw_response, variant.source)
        correct = extracted == variant.answer

        model_answers[level_name] = extracted
        expected_answers[level_name] = variant.answer
        correctness[level_name] = correct

        if verbose:
            status = "✓" if correct else "✗"
            print(f"  [{level_name}] {status} "
                  f"model={extracted} expected={variant.answer}")

        time.sleep(2)  # throttle to avoid Groq's 6000 TPM limit

    return SampleResult(
        sample_id=sample_id,
        source=sample.source,
        original_answer=sample.answer,
        model_answers=model_answers,
        expected_answers=expected_answers,
        correctness=correctness
    )


def _query_with_retry(variant, model_name, max_retries=3):
    """Query model with automatic retry on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return query_model(variant, model=model_name)
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait_time = 60
                print(f"  Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    return ""


# ============================================================
# AGGREGATION — turn per-sample results into contamination score
# ============================================================

def compute_contamination_score(
    results: List[SampleResult],
    model_name: str,
    benchmark: str
) -> ContaminationScore:
    """
    Aggregate per-sample correctness into per-level accuracy,
    then compute accuracy gap and contamination index.
    """
    levels = ["original", "L1_surface", "L2_numbers",
              "L3_structural", "L4_adversarial", "L5_compositional"]

    accuracy_by_level = {}
    for level in levels:
        n_correct = sum(1 for r in results if r.correctness.get(level, False))
        accuracy_by_level[level] = n_correct / len(results)

    original_acc = accuracy_by_level["original"]
    accuracy_gap = {
        level: original_acc - accuracy_by_level[level]
        for level in levels
        if level != "original"
    }

    # contamination index: average accuracy gap across perturbation levels
    # higher gap = more contamination
    contamination_index = sum(accuracy_gap.values()) / len(accuracy_gap)

    return ContaminationScore(
        model_name=model_name,
        benchmark=benchmark,
        n_samples=len(results),
        accuracy_by_level=accuracy_by_level,
        accuracy_gap=accuracy_gap,
        contamination_index=contamination_index
    )


# ============================================================
# REPORTING
# ============================================================

def print_contamination_report(score: ContaminationScore):
    """Pretty-print a contamination score."""
    print(f"\n{'=' * 60}")
    print(f"Contamination report: {score.model_name} on {score.benchmark}")
    print(f"{'=' * 60}")
    print(f"Samples evaluated: {score.n_samples}\n")

    print(f"{'Level':<20} {'Accuracy':>10} {'Gap from original':>20}")
    print(f"{'-' * 60}")
    for level, acc in score.accuracy_by_level.items():
        if level == "original":
            print(f"{level:<20} {acc:>10.2%} {'—':>20}")
        else:
            gap = score.accuracy_gap[level]
            print(f"{level:<20} {acc:>10.2%} {gap:>+20.2%}")

    print(f"\nContamination index: {score.contamination_index:.3f}")
    print(f"  (higher = more contamination signal)")


# ============================================================
# TEST HARNESS
# ============================================================

if __name__ == "__main__":
    MODEL = "llama-3.1-8b-instant"
    N_SAMPLES = 5   # keep small for testing — increase for real runs

    print(f"Running contamination evaluation...")
    print(f"Model: {MODEL}")
    print(f"Benchmark: GSM8K")
    print(f"Samples: {N_SAMPLES}\n")

    samples = load_gsm8k(n=N_SAMPLES)
    results = []

    for i, sample in enumerate(samples, start=1):
        result = evaluate_sample(sample, sample_id=i, model_name=MODEL, verbose=True)
        results.append(result)

    score = compute_contamination_score(results, model_name=MODEL, benchmark="gsm8k")
    print_contamination_report(score)