"""Pytest configuration shared across the test suite."""

from __future__ import annotations


def pytest_configure(config) -> None:  # type: ignore[no-untyped-def]
    config.addinivalue_line(
        "markers",
        "slow: marks tests that hit external APIs or download datasets",
    )