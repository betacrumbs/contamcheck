import os
import re
import json
from groq import Groq
from dotenv import load_dotenv
from src.benchmark import BenchmarkSample

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def _llm_call(prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

def _parse_json_response(response: str) -> dict:
    """LLMs sometimes wrap JSON in markdown — strip it out."""
    cleaned = re.sub(r'```json\s*|\s*```', '', response).strip()
    # find first { and last } to handle preamble text
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1:
        cleaned = cleaned[start:end+1]
    return json.loads(cleaned)


def level_1_surface(sample: BenchmarkSample) -> BenchmarkSample:
    """
    LEVEL 1 — Surface paraphrase.
    Same numbers, same logic, different wording. The easiest perturbation.
    A model with even shallow understanding should pass this.
    Failure here strongly suggests pure verbatim memorization.
    """
    prompt = (
        f"Rewrite this problem with significantly different vocabulary and sentence structure. "
        f"Keep all numbers and the answer identical. "
        f"Use synonyms aggressively, change sentence order, vary phrasing. "
        f"Return ONLY the rewritten problem.\n\n"
        f"Problem: {sample.question}"
    )
    return BenchmarkSample(
        question=_llm_call(prompt),
        answer=sample.answer,
        domain=sample.domain,
        source=sample.source
    )


def level_2_number_swap(sample: BenchmarkSample) -> BenchmarkSample:
    """
    LEVEL 2 — Number perturbation with recomputed answer.
    Replace every number in the question with a different number,
    then have the LLM recompute the new correct answer.
    A memorizing model that learned 'this question shape -> 18' will fail
    because the new answer isn't 18 anymore.
    """
    prompt = (
        f"Take this math problem and rewrite it by changing every number to a different number "
        f"(keep them reasonable — same order of magnitude). Then solve the new problem and give the answer.\n\n"
        f"Original problem: {sample.question}\n"
        f"Original answer: {sample.answer}\n\n"
        f"Return ONLY a JSON object with this exact format:\n"
        f'{{"question": "the new problem text", "answer": "the new numerical answer"}}'
    )
    try:
        result = _parse_json_response(_llm_call(prompt, temperature=0.3))
        return BenchmarkSample(
            question=result["question"],
            answer=str(result["answer"]),
            domain=sample.domain,
            source=sample.source
        )
    except (json.JSONDecodeError, KeyError):
        # fallback if LLM output malformed — return original so pipeline doesn't crash
        return sample


def level_3_structural_rewrite(sample: BenchmarkSample) -> BenchmarkSample:
    """
    LEVEL 3 — Structural rewrite.
    Same numbers, same final answer, but the reasoning path is restructured.
    Examples: invert the question (ask 'how many used' instead of 'how many left'),
    split into multiple sub-steps, combine steps, change perspective.
    A memorizing model fails because the answer pattern doesn't match the
    new question structure.
    """
    prompt = (
        f"Rewrite this math problem so the final answer stays exactly the same, "
        f"but the reasoning structure is different. Use one of these techniques:\n"
        f"- Invert the question (e.g. ask what was used vs what's left)\n"
        f"- Reorder the information so the steps appear in a different sequence\n"
        f"- Reframe from a different perspective (e.g. third person, different character)\n"
        f"- Combine or split sub-calculations\n\n"
        f"The numbers and final answer must remain identical.\n\n"
        f"Original problem: {sample.question}\n"
        f"Answer must remain: {sample.answer}\n\n"
        f"Return ONLY the rewritten problem."
    )
    return BenchmarkSample(
        question=_llm_call(prompt, temperature=0.6),
        answer=sample.answer,
        domain=sample.domain,
        source=sample.source
    )


def level_4_adversarial(sample: BenchmarkSample) -> BenchmarkSample:
    """
    LEVEL 4 — Adversarial perturbation.
    Add a distractor — irrelevant numerical information that a memorizing
    or pattern-matching model would incorrectly incorporate. A reasoning
    model recognizes the distractor is unrelated and ignores it.
    Also reskin into a completely unfamiliar domain.
    This is the strongest test — failure here is normal even for some
    reasoning models, but a contaminated model fails dramatically.
    """
    prompt = (
        f"Rewrite this math problem with three modifications:\n"
        f"1. Move it into a completely unfamiliar domain (deep sea biology, "
        f"medieval blacksmithing, asteroid mining — pick something unusual).\n"
        f"2. Add ONE irrelevant numerical fact that should NOT affect the answer "
        f"(e.g. 'the workshop has been operating for 47 years' — a number that "
        f"a careless solver might multiply or add).\n"
        f"3. Keep the actual math operations and the final answer EXACTLY the same.\n\n"
        f"Original problem: {sample.question}\n"
        f"Answer must remain: {sample.answer}\n\n"
        f"Return ONLY the rewritten problem."
    )
    return BenchmarkSample(
        question=_llm_call(prompt, temperature=0.7, max_tokens=400),
        answer=sample.answer,
        domain=sample.domain,
        source=sample.source
    )


def generate_all_perturbations(sample: BenchmarkSample) -> dict:
    """
    Returns the original plus all 4 graduated perturbation levels.
    Each level tests a different aspect of memorization vs reasoning.
    """
    return {
        "original":      sample,
        "L1_surface":    level_1_surface(sample),
        "L2_numbers":    level_2_number_swap(sample),
        "L3_structural": level_3_structural_rewrite(sample),
        "L4_adversarial": level_4_adversarial(sample)
    }


if __name__ == "__main__":
    from src.benchmark import load_gsm8k

    print("Testing graduated perturbation engine on 2 GSM8K samples...\n")
    samples = load_gsm8k(n=2)

    for i, sample in enumerate(samples):
        print(f"{'='*70}")
        print(f"SAMPLE {i+1}")
        print(f"{'='*70}")
        variants = generate_all_perturbations(sample)
        for variant_name, variant in variants.items():
            print(f"\n[{variant_name.upper()}]")
            print(f"  Q: {variant.question}")
            print(f"  A: {variant.answer}")
        print()