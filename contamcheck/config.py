"""Central registry of constants for contamcheck.

Nothing else in the codebase should hardcode model names, default seeds,
or supported access tiers. If you find yourself writing a string literal
for a model name in another file, add it here instead.
"""

from __future__ import annotations

from typing import Literal

# Access tiers, ordered from least to most access.
# A signal that requires "logprobs" can also run on "weights" but not "api".
AccessTier = Literal["api", "logprobs", "weights"]
ACCESS_TIERS: tuple[AccessTier, ...] = ("api", "logprobs", "weights")

# Perturbation type codes. See DESIGN.md for the full taxonomy.
PerturbationType = Literal["P", "N", "S", "T", "A"]
PERTURBATION_TYPES: tuple[PerturbationType, ...] = ("P", "N", "S", "T", "A")

# Default models for development. Change these in one place, not five.
DEFAULT_GENERATOR_MODEL = "google/gemma-4-31b-it:free"
DEFAULT_EVALUATOR_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# OpenRouter is the primary API gateway. The base URL is OpenAI-compatible.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default seed for any operation that needs one. Audits require explicit seeds;
# this is only used as a fallback for internal utilities and tests.
DEFAULT_SEED = 42

# Supported benchmarks. Add to this set as we implement loaders for each.
SUPPORTED_BENCHMARKS: frozenset[str] = frozenset({"gsm8k"})