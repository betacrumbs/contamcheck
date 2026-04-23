from dataclasses import dataclass
from datasets import load_dataset
from typing import List

@dataclass
class BenchmarkSample:
    question: str
    answer: str
    domain: str
    source: str

def load_gsm8k(n: int = 200) -> List[BenchmarkSample]:
    ds = load_dataset("gsm8k", "main", split="test")
    samples = []
    for item in ds.select(range(n)):
        samples.append(BenchmarkSample(
            question=item["question"],
            answer=item["answer"].split("####")[-1].strip(),
            domain="math",
            source="gsm8k"
        ))
    return samples

def load_mmlu(n: int = 200) -> List[BenchmarkSample]:
    ds = load_dataset("cais/mmlu", "all", split="test")
    samples = []
    choices_labels = ["A", "B", "C", "D"]
    for item in ds.select(range(n)):
        choices_text = "\n".join(
            f"{choices_labels[i]}. {c}"
            for i, c in enumerate(item["choices"])
        )
        question = f"{item['question']}\n{choices_text}"
        answer = choices_labels[item["answer"]]
        samples.append(BenchmarkSample(
            question=question,
            answer=answer,
            domain=item["subject"],
            source="mmlu"
        ))
    return samples

def load_math(n: int = 200) -> List[BenchmarkSample]:
    ds = load_dataset("EleutherAI/hendrycks_math", "algebra", split="test")
    samples = []
    for item in ds.select(range(n)):
        samples.append(BenchmarkSample(
            question=item["problem"],
            answer=item["solution"],
            domain="algebra",
            source="math"
        ))
    return samples

def load_humaneval(n: int = 164) -> List[BenchmarkSample]:
    ds = load_dataset("openai/openai_humaneval", split="test")
    n = min(n, len(ds))  # can't load more than what exists
    samples = []
    for item in ds.select(range(n)):
        samples.append(BenchmarkSample(
            question=item["prompt"],
            answer=item["canonical_solution"],
            domain="coding",
            source="humaneval"
        ))
    return samples

def load_all(n: int = 200) -> List[BenchmarkSample]:
    print("Loading all benchmarks...")
    all_samples = []
    loaders = [
        ("GSM8K", load_gsm8k),
        ("MMLU", load_mmlu),
        ("MATH", load_math),
        ("HumanEval", load_humaneval),
    ]
    for name, loader in loaders:
        print(f"  Loading {name}...")
        samples = loader(n=n)
        print(f"  Loaded {len(samples)} samples from {name}")
        all_samples.extend(samples)
    print(f"\nTotal: {len(all_samples)} samples across all benchmarks")
    return all_samples

if __name__ == "__main__":
    samples = load_all(n=5)
    print("\n--- Preview ---")
    for s in samples:
        print(f"[{s.source.upper()}] {s.question[:80]}...")
        print(f"  Answer: {s.answer[:50]}")
        print()