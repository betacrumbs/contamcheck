import os
import re
import json
from typing import Tuple
from groq import Groq
from dotenv import load_dotenv
from src.benchmark import BenchmarkSample

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ============================================================
# CORE LLM AND PARSING UTILITIES
# ============================================================

def _llm_call(prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
    """Single point of contact with the Groq API."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def _solve_problem(question: str) -> str:
    """
    Independently solve a generated problem to verify the answer.
    Used by levels that change the answer (L2, L5).
    Returns the extracted numerical answer as a string, or empty if extraction fails.
    """
    prompt = (
        f"Solve this math problem step by step. "
        f"On the final line write exactly: Answer: <number>\n\n{question}"
    )
    response = _llm_call(prompt, temperature=0, max_tokens=400)
    match = re.search(r'Answer:\s*\$?([\d,]+(?:\.\d+)?)', response)
    return match.group(1).replace(",", "") if match else ""


# ============================================================
# CENTRALIZED VALIDATION
# ============================================================

def _is_integer_answer(answer: str) -> bool:
    """True if answer string represents a whole number."""
    try:
        val = float(answer)
        return val == int(val)
    except (ValueError, TypeError):
        return False


def _contains_solution(text: str) -> bool:
    """True if text contains calculation working like '= 18' or '* 2 ='."""
    return bool(
        re.search(r'\d\s*=\s*\d', text) or
        re.search(r'[\+\-\*\/]\s*\d+\s*=', text)
    )


def _word_overlap(a: str, b: str) -> float:
    """Fraction of words from a that also appear in b."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a:
        return 0.0
    return len(words_a & words_b) / len(words_a)


def is_valid_perturbation(
    candidate: BenchmarkSample,
    original: BenchmarkSample,
    expects_same_answer: bool
) -> Tuple[bool, str]:
    """
    Single source of truth for perturbation quality.
    Every check applies uniformly to every level.

    Returns (is_valid, reason). Reason is a short diagnostic string
    used for logging when a perturbation fails validation.
    """
    q = (candidate.question or "").strip()
    a = (candidate.answer or "").strip()

    if not q:
        return False, "empty question"

    if q == original.question.strip():
        return False, "question identical to original"

    if not q.endswith("?"):
        return False, "question does not end with '?'"

    if _contains_solution(q):
        return False, "solution working leaked into question"

    if a in q:
        return False, f"answer '{a}' appears verbatim in question"

    if not _is_integer_answer(a):
        return False, f"answer '{a}' is not an integer"

    if expects_same_answer and a != original.answer:
        return False, f"answer changed ({original.answer} -> {a}) when it should not have"

    if not expects_same_answer and a == original.answer:
        return False, "answer unchanged when it should have changed"

    overlap = _word_overlap(original.question, q)
    if overlap > 0.75:
        return False, f"too similar to original ({overlap:.0%} word overlap)"

    return True, "ok"


# ============================================================
# PERTURBATION LEVELS — single attempt per level
# Each function returns ONE candidate. Validation and retries
# are handled centrally in generate_all_perturbations.
# ============================================================

def level_1_surface(sample: BenchmarkSample) -> BenchmarkSample:
    """
    L1 — Surface paraphrase.
    Same numbers, same answer, completely different wording.
    Tests resistance to verbatim text memorization.
    """
    prompt = (
        f"Rewrite this problem using completely different vocabulary and sentence structure. "
        f"Keep all numbers identical. The answer must remain {sample.answer}.\n\n"
        f"Strict rules:\n"
        f"- Do not show calculations, working, or equals signs\n"
        f"- Do not mention the answer ({sample.answer}) anywhere\n"
        f"- End with a single question mark\n"
        f"- Return ONLY the rewritten question, nothing else\n\n"
        f"Problem: {sample.question}"
    )
    return BenchmarkSample(
        question=_llm_call(prompt, temperature=0.7),
        answer=sample.answer,
        domain=sample.domain,
        source=sample.source
    )


def level_2_number_swap(sample: BenchmarkSample) -> BenchmarkSample:
    """
    L2 — Number swap with recomputed answer.
    Different numbers throughout, new answer computed by independent solve.
    Tests resistance to memorized 'this question shape -> this answer'.
    """
    gen_prompt = (
        f"Rewrite this math problem changing every number to a different whole number "
        f"of similar magnitude. Change names if present.\n\n"
        f"Strict arithmetic rules:\n"
        f"- Use ONLY simple ratios: 'half', 'double', 'triple'\n"
        f"- If you use 'half', the base number must be even\n"
        f"- If you use 'triple' or higher, all combinations must yield whole numbers\n"
        f"- Every intermediate calculation must produce a whole number\n"
        f"- The final answer must be a whole number\n\n"
        f"Output rules:\n"
        f"- End with a question mark\n"
        f"- Do not include the answer or any calculations in the question\n"
        f"- Return ONLY the rewritten problem\n\n"
        f"Original: {sample.question}"
    )
    new_question = _llm_call(gen_prompt, temperature=0.3, max_tokens=300)
    new_answer = _solve_problem(new_question) or sample.answer
    return BenchmarkSample(
        question=new_question,
        answer=new_answer,
        domain=sample.domain,
        source=sample.source
    )


def level_3_structural_rewrite(sample: BenchmarkSample) -> BenchmarkSample:
    """
    L3 — Structural rewrite.
    Same numbers, same answer, different reasoning path.
    Tests resistance to memorized answer-shape patterns.
    """
    prompt = (
        f"Rewrite this math problem so the final answer is unchanged "
        f"but the reasoning structure is different. Use one technique:\n"
        f"- Invert the question (ask what was used vs what's left)\n"
        f"- Reorder the information so steps appear in different sequence\n"
        f"- Reframe from a different perspective or character\n\n"
        f"Strict rules:\n"
        f"- The answer must remain {sample.answer} but must NOT be mentioned in the question\n"
        f"- No calculations, working, or equals signs\n"
        f"- End with a question mark\n"
        f"- Return ONLY the rewritten question\n\n"
        f"Original: {sample.question}"
    )
    return BenchmarkSample(
        question=_llm_call(prompt, temperature=0.6, max_tokens=400),
        answer=sample.answer,
        domain=sample.domain,
        source=sample.source
    )


def level_4_adversarial(sample: BenchmarkSample) -> BenchmarkSample:
    """
    L4 — Adversarial reskin with distractor.
    Unfamiliar domain plus one irrelevant numerical distractor.
    Tests resistance to surface pattern matching and number greediness.
    """
    prompt = (
        f"Rewrite this math problem with two changes:\n"
        f"1. Move it to a completely unfamiliar domain "
        f"(deep sea biology, medieval blacksmithing, asteroid mining, ancient astronomy).\n"
        f"2. Add ONE clearly irrelevant numerical distractor that does not affect the calculation "
        f"(e.g. 'the workshop has stood for 23 years', 'the chamber holds 412 cubic meters').\n\n"
        f"Strict rules:\n"
        f"- The actual math operations and final answer must remain EXACTLY {sample.answer}\n"
        f"- Do not mention the answer ({sample.answer}) anywhere in the question\n"
        f"- The distractor number must be obviously unrelated to the calculation\n"
        f"- No calculations or equals signs\n"
        f"- End with a question mark\n"
        f"- Return ONLY the rewritten question\n\n"
        f"Original: {sample.question}"
    )
    return BenchmarkSample(
        question=_llm_call(prompt, temperature=0.7, max_tokens=400),
        answer=sample.answer,
        domain=sample.domain,
        source=sample.source
    )


def level_5_compositional(sample: BenchmarkSample) -> BenchmarkSample:
    """
    L5 — Compositional perturbation (the strongest test).
    Number swap + adversarial reskin + distractor, simultaneously.
    Answer is independently verified after generation.
    Tests true cross-domain reasoning, distractor robustness, and recomputation.
    """
    gen_prompt = (
        f"Rewrite this math problem with ALL these changes simultaneously:\n"
        f"1. Move to a completely unfamiliar domain "
        f"(deep sea biology, medieval blacksmithing, asteroid mining, ancient astronomy).\n"
        f"2. Change every number to a different whole number.\n"
        f"3. Add ONE clearly irrelevant numerical distractor.\n\n"
        f"Strict arithmetic rules:\n"
        f"- Use ONLY simple ratios: 'half', 'double', 'triple'\n"
        f"- If you use 'half', the base number must be even\n"
        f"- Every intermediate calculation must produce a whole number\n"
        f"- The final answer must be a whole number\n\n"
        f"Output rules:\n"
        f"- End with a question mark\n"
        f"- No calculations, working, or equals signs\n"
        f"- Return ONLY the rewritten problem\n\n"
        f"Original: {sample.question}"
    )
    new_question = _llm_call(gen_prompt, temperature=0.5, max_tokens=400)
    new_answer = _solve_problem(new_question) or sample.answer
    return BenchmarkSample(
        question=new_question,
        answer=new_answer,
        domain=sample.domain,
        source=sample.source
    )


# ============================================================
# ORCHESTRATION — central retry and validation
# ============================================================

LEVEL_DEFINITIONS = [
    ("L1_surface",      level_1_surface,         True),
    ("L2_numbers",      level_2_number_swap,     False),
    ("L3_structural",   level_3_structural_rewrite, True),
    ("L4_adversarial",  level_4_adversarial,     True),
    ("L5_compositional", level_5_compositional,  False),
]

MAX_ATTEMPTS = 3


def generate_all_perturbations(sample: BenchmarkSample, verbose: bool = False) -> dict:
    """
    Generate all 5 perturbation levels with centralized validation and retries.

    Each level gets up to MAX_ATTEMPTS attempts. The first valid candidate is kept.
    If all attempts fail validation, the original sample is used as fallback and
    a warning is printed with the failure reason.

    Returns dict with keys: original, L1_surface, L2_numbers, L3_structural,
    L4_adversarial, L5_compositional.
    """
    results = {"original": sample}

    for level_name, level_fn, expects_same_answer in LEVEL_DEFINITIONS:
        last_reason = "no attempts made"

        for attempt in range(1, MAX_ATTEMPTS + 1):
            candidate = level_fn(sample)
            valid, reason = is_valid_perturbation(
                candidate, sample, expects_same_answer
            )
            if valid:
                results[level_name] = candidate
                if verbose:
                    print(f"  {level_name}: ok (attempt {attempt})")
                break
            last_reason = reason
            if verbose:
                print(f"  {level_name}: attempt {attempt} rejected — {reason}")
        else:
            print(f"  WARNING: {level_name} fell back to original "
                  f"after {MAX_ATTEMPTS} attempts (last reason: {last_reason})")
            results[level_name] = sample

    return results


# ============================================================
# TEST HARNESS
# ============================================================

if __name__ == "__main__":
    from src.benchmark import load_gsm8k

    print("Testing 5-level perturbation engine on 2 GSM8K samples...\n")
    samples = load_gsm8k(n=2)

    for i, sample in enumerate(samples, start=1):
        print(f"{'=' * 70}")
        print(f"SAMPLE {i}")
        print(f"{'=' * 70}\n")
        variants = generate_all_perturbations(sample, verbose=True)
        for name, variant in variants.items():
            print(f"\n[{name.upper()}]")
            print(f"  Q: {variant.question}")
            print(f"  A: {variant.answer}")
        print()