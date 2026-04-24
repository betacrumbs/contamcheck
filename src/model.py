import os
import re
from groq import Groq
from dotenv import load_dotenv
from src.benchmark import BenchmarkSample

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def query_model(sample: BenchmarkSample, model: str = "llama-3.1-8b-instant") -> str:
    prompt = build_prompt(sample)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1024   # was 512 — small models need room to reason
    )
    return response.choices[0].message.content.strip()

def extract_answer(response: str, source: str) -> str:
    if not response or not response.strip():
        return ""

    # Strategy 1: explicit "Answer:" line — most reliable
    match = re.search(r'Answer:\s*\$?([\d,]+(?:\.\d+)?)', response, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")

    # Strategy 2: MMLU letter at end of response
    if source == "mmlu":
        tail = response[-100:]
        match = re.search(r'\b([A-D])\b(?!.*\b[A-D]\b)', tail)
        if match:
            return match.group(1)
        match = re.search(r'\b([A-D])\b', response)
        if match:
            return match.group(1)

    # Strategy 3: common answer-indicator patterns
    patterns = [
        r'(?:the answer is|answer is|equals|total is|result is|makes)\s*\$?([\d,]+(?:\.\d+)?)',
        r'=\s*\$?([\d,]+(?:\.\d+)?)\s*(?:\.|$|\n)',
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "")

    # Strategy 4: last number in response (math answers usually end with the number)
    numbers = re.findall(r'\$?([\d,]+(?:\.\d+)?)', response)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""

def build_prompt(sample: BenchmarkSample) -> str:
    if sample.source == "humaneval":
        return (
            f"Complete the following Python function. "
            f"Return only the code, no explanation.\n\n{sample.question}"
        )
    elif sample.source == "mmlu":
        return (
            f"Answer the following multiple choice question. "
            f"Think step by step, then end with 'Answer: X' where X is A, B, C, or D.\n\n"
            f"{sample.question}"
        )
    else:
        return (
            f"Solve the following problem step by step. "
            f"Show your reasoning clearly. "
            f"IMPORTANT: Your response MUST end with exactly this line:\n"
            f"Answer: N\n"
            f"where N is a single integer (no units, no dollar signs, no commas, no decimals). "
            f"Do not continue reasoning after the Answer line.\n\n"
            f"{sample.question}"
        )

if __name__ == "__main__":
    from src.benchmark import load_gsm8k

    print("Testing model querying on 3 GSM8K samples...\n")
    samples = load_gsm8k(n=3)

    for i, sample in enumerate(samples):
        print(f"--- Sample {i+1} ---")
        print(f"Question: {sample.question[:100]}...")
        print(f"Expected: {sample.answer}")
        response = query_model(sample)
        extracted = extract_answer(response, sample.source)
        correct = extracted == sample.answer
        print(f"Model:    {extracted}")
        print(f"Correct:  {correct}")
        print()