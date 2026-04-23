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

if __name__ == "__main__":
    print("Loading GSM8K...")
    samples = load_gsm8k(n=5)
    print(f"Loaded {len(samples)} samples\n")
    for i, s in enumerate(samples):
        print(f"--- Sample {i+1} ---")
        print(f"Question: {s.question[:100]}...")
        print(f"Answer: {s.answer}")
        print()