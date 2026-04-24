import os
import re
import random
from typing import Tuple, Optional
from groq import Groq
from dotenv import load_dotenv
from src.benchmark import BenchmarkSample

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ============================================================
# LLM UTILITIES
# ============================================================

def _llm_call(prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


# ============================================================
# NUMBER EXTRACTION — lets us do arithmetic in Python, not LLM
# ============================================================

def _extract_numbers(text: str) -> list:
    """Extract all integers from text in order of appearance."""
    return [int(m) for m in re.findall(r'\b\d+\b', text)]


def _substitute_numbers(text: str, old_numbers: list, new_numbers: list) -> str:
    """Replace each number in text with its corresponding new number, in order."""
    result = text
    # replace from right to left to avoid index issues
    for old, new in reversed(list(zip(old_numbers, new_numbers))):
        # word-boundary match to avoid replacing parts of larger numbers
        result = re.sub(rf'\b{old}\b', str(new), result, count=1)
    return result


# ============================================================
# VALIDATION — simpler and less brittle
# ============================================================

def _is_integer_answer(answer: str) -> bool:
    try:
        val = float(answer)
        return val == int(val)
    except (ValueError, TypeError):
        return False


def _contains_solution(text: str) -> bool:
    return bool(
        re.search(r'\d\s*=\s*\d', text) or
        re.search(r'[\+\-\*\/]\s*\d+\s*=', text)
    )


def _answer_appears_isolated(answer: str, question: str) -> bool:
    """
    Check if answer appears as an isolated number in the question
    (not as part of a larger number or date).
    This avoids false positives like '18' matching '1826'.
    """
    # only trigger if answer is 2+ digits to avoid noise from '0', '1', '2', etc.
    if len(answer) < 2:
        return False
    return bool(re.search(rf'\b{re.escape(answer)}\b', question))


def is_valid_perturbation(
    candidate: BenchmarkSample,
    original: BenchmarkSample,
    expects_same_answer: bool
) -> Tuple[bool, str]:
    q = (candidate.question or "").strip()
    a = (candidate.answer or "").strip()

    if not q:
        return False, "empty question"

    if q == original.question.strip():
        return False, "question identical to original"

    if "?" not in q[-15:]:
        return False, "question does not end with '?'"

    if _contains_solution(q):
        return False, "solution working leaked into question"

    if _answer_appears_isolated(a, q):
        return False, f"answer '{a}' appears isolated in question"

    if not _is_integer_answer(a):
        return False, f"answer '{a}' is not an integer"

    if expects_same_answer and a != original.answer:
        return False, f"answer changed ({original.answer} -> {a})"

    if not expects_same_answer and a == original.answer:
        return False, "answer unchanged when it should have changed"

    # same-answer levels must preserve all numerical information
    if expects_same_answer:
        original_numbers = set(_extract_numbers(original.question))
        candidate_numbers = set(_extract_numbers(q))
        missing = original_numbers - candidate_numbers
        if missing:
            return False, f"dropped numbers from original: {missing}"

    return True, "ok"

# ============================================================
# PERTURBATION LEVELS
# ============================================================

def level_1_surface(sample: BenchmarkSample) -> BenchmarkSample:
    """L1 — Surface paraphrase. Same numbers, same answer."""
    prompt = (
        f"Rewrite this problem with completely different vocabulary and sentence structure. "
        f"Keep every number identical. The answer must remain {sample.answer} "
        f"but do not mention {sample.answer} anywhere.\n"
        f"No calculations or equals signs. End with a question mark.\n\n"
        f"Problem: {sample.question}\n\n"
        f"Return ONLY the rewritten question."
    )
    return BenchmarkSample(
        question=_llm_call(prompt, temperature=0.7),
        answer=sample.answer,
        domain=sample.domain,
        source=sample.source
    )


def level_2_number_swap(sample: BenchmarkSample) -> BenchmarkSample:
    """
    L2 — Number swap.
    Programmatically pick new numbers (scaled by 1.5-3x), substitute them in,
    then independently solve the new problem. This removes LLM arithmetic risk.
    """
    original_numbers = _extract_numbers(sample.question)
    if not original_numbers:
        return sample

    # scale numbers by random integer multiplier, keeping whole-number math
    multiplier = random.choice([2, 3])
    new_numbers = [n * multiplier for n in original_numbers]

    new_question = _substitute_numbers(sample.question, original_numbers, new_numbers)

    # solve independently — most answer operations scale linearly with multiplier
    # so the new answer should be multiplier * original_answer for most GSM8K problems
    # but we verify with the LLM to be safe
    solve_prompt = (
        f"Solve step by step. On the final line write: Answer: <integer>\n\n{new_question}"
    )
    solve_response = _llm_call(solve_prompt, temperature=0, max_tokens=400)
    match = re.search(r'Answer:\s*\$?([\d,]+)', solve_response)
    new_answer = match.group(1).replace(",", "") if match else ""

    if not new_answer or not _is_integer_answer(new_answer):
        return sample

    return BenchmarkSample(
        question=new_question,
        answer=new_answer,
        domain=sample.domain,
        source=sample.source
    )


def level_3_structural_rewrite(sample: BenchmarkSample) -> BenchmarkSample:
    """L3 — Same numbers, same answer, different reasoning path."""
    original_numbers = _extract_numbers(sample.question)
    prompt = (
        f"Rewrite this problem so the answer stays {sample.answer} but the reasoning "
        f"structure is different. Techniques: invert the question, reorder information, "
        f"change perspective.\n\n"
        f"CRITICAL: The rewritten question must contain ALL of these numbers: {original_numbers}. "
        f"Every number from the original must appear somewhere in the rewrite. "
        f"You may rephrase but cannot omit any number.\n\n"
        f"Do not mention {sample.answer}. No calculations. End with a question mark.\n\n"
        f"Problem: {sample.question}\n\n"
        f"Return ONLY the rewritten question."
    )
    return BenchmarkSample(
        question=_llm_call(prompt, temperature=0.6, max_tokens=400),
        answer=sample.answer,
        domain=sample.domain,
        source=sample.source
    )


def level_4_adversarial(sample: BenchmarkSample) -> BenchmarkSample:
    """L4 — Unfamiliar domain + one distractor. Same numbers, same answer."""
    prompt = (
        f"Rewrite this problem with two changes:\n"
        f"1. Move it to an unfamiliar domain (deep sea biology, medieval blacksmithing, "
        f"asteroid mining, ancient astronomy).\n"
        f"2. Add ONE irrelevant numerical distractor in its own sentence.\n\n"
        f"CRITICAL: Keep the exact same math operations (same additions, subtractions, "
        f"multiplications). The answer must remain {sample.answer}.\n"
        f"Do not mention {sample.answer}. No calculations. End with a question mark.\n\n"
        f"Problem: {sample.question}\n\n"
        f"Return ONLY the rewritten question."
    )
    return BenchmarkSample(
        question=_llm_call(prompt, temperature=0.7, max_tokens=400),
        answer=sample.answer,
        domain=sample.domain,
        source=sample.source
    )


def level_5_compositional(sample: BenchmarkSample) -> BenchmarkSample:
    """
    L5 — Programmatic number swap + LLM-based adversarial reskin + distractor.
    Numbers change programmatically. The LLM only handles creative reskinning
    of the already-number-swapped question.
    """
    # step 1 — programmatic number swap (same approach as L2)
    original_numbers = _extract_numbers(sample.question)
    if not original_numbers:
        return sample
    multiplier = random.choice([2, 3])
    new_numbers = [n * multiplier for n in original_numbers]
    number_swapped_q = _substitute_numbers(sample.question, original_numbers, new_numbers)

    # step 2 — solve the number-swapped question to get the new correct answer
    solve_prompt = (
        f"Solve step by step. On the final line write: Answer: <integer>\n\n{number_swapped_q}"
    )
    solve_response = _llm_call(solve_prompt, temperature=0, max_tokens=400)
    match = re.search(r'Answer:\s*\$?([\d,]+)', solve_response)
    new_answer = match.group(1).replace(",", "") if match else ""
    if not new_answer or not _is_integer_answer(new_answer):
        return sample

    # step 3 — now ask LLM to reskin into a new domain with a distractor,
    # preserving the numbers we already chose
    reskin_prompt = (
        f"Rewrite this math problem with two changes:\n"
        f"1. Move to an unfamiliar domain (deep sea biology, medieval blacksmithing, "
        f"asteroid mining, ancient astronomy).\n"
        f"2. Add ONE irrelevant numerical distractor in its own sentence.\n\n"
        f"CRITICAL RULES:\n"
        f"- Keep ALL numbers from the current version exactly as they are: {new_numbers}\n"
        f"- Do not change the math operations\n"
        f"- Do not mention {new_answer} anywhere\n"
        f"- No calculations. End with a question mark.\n\n"
        f"Current version: {number_swapped_q}\n\n"
        f"Return ONLY the rewritten question."
    )
    reskinned_q = _llm_call(reskin_prompt, temperature=0.6, max_tokens=400)

    return BenchmarkSample(
        question=reskinned_q,
        answer=new_answer,
        domain=sample.domain,
        source=sample.source
    )


# ============================================================
# ORCHESTRATION
# ============================================================

LEVEL_DEFINITIONS = [
    ("L1_surface",       level_1_surface,              True),
    ("L2_numbers",       level_2_number_swap,          False),
    ("L3_structural",    level_3_structural_rewrite,   True),
    ("L4_adversarial",   level_4_adversarial,          True),
    ("L5_compositional", level_5_compositional,        False),
]

MAX_ATTEMPTS = 3


def generate_all_perturbations(sample: BenchmarkSample, verbose: bool = False) -> dict:
    results = {"original": sample}

    for level_name, level_fn, expects_same_answer in LEVEL_DEFINITIONS:
        last_reason = "no attempts"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            candidate = level_fn(sample)
            valid, reason = is_valid_perturbation(candidate, sample, expects_same_answer)
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
                  f"after {MAX_ATTEMPTS} attempts (last: {last_reason})")
            results[level_name] = sample

    return results


if __name__ == "__main__":
    from src.benchmark import load_gsm8k

    print("Testing 5-level perturbation engine on 2 GSM8K samples...\n")
    samples = load_gsm8k(n=2)

    for i, sample in enumerate(samples, start=1):
        print(f"{'=' * 70}\nSAMPLE {i}\n{'=' * 70}\n")
        variants = generate_all_perturbations(sample, verbose=True)
        for name, variant in variants.items():
            print(f"\n[{name.upper()}]")
            print(f"  Q: {variant.question}")
            print(f"  A: {variant.answer}")
        print()