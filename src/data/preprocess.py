"""Shared, side-effect-free transformations for dataset-building scripts."""

from __future__ import annotations

import re


def replace_int_with_c(tokens):
    """Replace positive integer tokens outside [-2, 2] with CONST buckets."""
    keep_values = set(range(-2, 3))
    normalized = list(tokens)
    for index, token in enumerate(tokens):
        if isinstance(token, str) and token.isdigit() and int(token) not in keep_values:
            if len(token) == 1:
                normalized[index] = "CONST1"
            elif len(token) == 2:
                normalized[index] = "CONST2"
            else:
                normalized[index] = "CONST3"
    return normalized


# Compatibility name used by older notebooks.
replace_int_with_C = replace_int_with_c


def contains_unsupported_expressions(expression: str) -> bool:
    """Return whether an expression contains an unsupported complex form."""
    abs_complex_pattern = r"(abs|Complex)\s*\([^)]*,[^)]*\)"
    complex_number_pattern = r"\b-?\d+\s*\+\s*-?(?:\d+\*)?I\b"
    return bool(re.search(abs_complex_pattern, expression)) or bool(
        re.search(complex_number_pattern, expression)
    )


def min_max_scale(labels):
    """Scale valid labels to [0, 1] while preserving the -1 missing marker."""
    valid = [value for value in labels if value != -1]
    if not valid:
        return list(labels)
    minimum = min(valid)
    value_range = max(valid) - minimum
    return [
        -1
        if value == -1
        else 0.0
        if value_range == 0
        else (value - minimum) / value_range
        for value in labels
    ]
