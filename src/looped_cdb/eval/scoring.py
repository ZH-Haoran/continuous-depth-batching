"""Scoring primitives: regex answer-extraction filters and an exact-match metric."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RegexFilter:
    """Extract an answer from generated text with a regex.

    ``group_select`` indexes into the list of ``re.findall`` matches (not into a
    single match's groups); ``-1`` therefore selects the last occurrence. When a
    match is a tuple (pattern with multiple groups), the first non-empty group is
    used. ``fallback`` is returned when nothing matches.
    """

    pattern: str
    group_select: int = 0
    fallback: str = "[invalid]"

    def apply(self, text: str) -> str:
        matches = re.findall(self.pattern, text)
        if not matches:
            return self.fallback
        match = matches[self.group_select]
        if isinstance(match, tuple):
            non_empty = [group for group in match if group]
            match = non_empty[0] if non_empty else ""
        return match.strip()


def exact_match(
    prediction: str,
    gold: str,
    *,
    regexes_to_ignore: list[str],
    ignore_case: bool = True,
) -> int:
    """Return 1 if prediction equals gold after normalization, else 0.

    Each pattern in ``regexes_to_ignore`` is stripped from both sides before
    comparison.
    """

    def normalize(text: str) -> str:
        for pattern in regexes_to_ignore:
            text = re.sub(pattern, "", text)
        text = text.strip()
        return text.lower() if ignore_case else text

    return int(normalize(prediction) == normalize(gold))
