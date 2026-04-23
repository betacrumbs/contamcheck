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
        max_tokens=512
    )
    return response.choices[0].message.content.strip()

def extract_answer(response: str, source: str) -> str:
    """
    Pulls the final answer out of the model's response.
    Handles dollar signs, commas, and the 'Answer:' prefix.
    """
    # look for explicit "Answer: ..." line first
    match = re.search(r'Answer:\s*\$?([\d,]+(?:\.\d+)?)', response)
    if match:
        return match.group(1).replace(",", "")
    
    # for MMLU — just look for a single letter A/B/C/D
    if source == "mmlu":
        match = re.search(r'\b([A-D])\b', response)
        if match:
            return match.group(1)
    
    # fallback — grab the last number in the response
    numbers = re.findall(r'\$?([\d,]+(?:\.\d+)?)', response)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return response.strip()

def build_prompt(sample: BenchmarkSample) -> str:
    if sample.source == "humaneval":
        return (
            f"Complete the following Python function. "
            f"Return only the code, no explanation.\n\n{sample.question}"
        )
    elif sample.source == "mmlu":
        return (
            f"Answer the following multiple choice question. "
            f"Reply with only the letter (A, B, C, or D).\n\n{sample.question}"
        )
    else:
        return (
            f"Solve the following problem step by step. "
            f"At the end, write your final answer on a new line starting with 'Answer:'\n\n"
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