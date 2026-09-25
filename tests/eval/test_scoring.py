"""Unit tests for the scoring primitives."""

from looped_cdb.eval.scoring import RegexFilter, exact_match

STRICT = RegexFilter(r"The answer is (\-?[0-9\.\,]+).", group_select=0)
FLEXIBLE = RegexFilter(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", group_select=-1)
IGNORE = [",", r"\$", r"(?s).*#### ", r"\.$"]


def test_strict_extracts_answer_phrase():
    assert STRICT.apply("...so 5 + 4 = 9. The answer is 9.") == "9"


def test_strict_returns_fallback_when_absent():
    assert STRICT.apply("no final line here") == "[invalid]"


def test_flexible_takes_last_match():
    assert FLEXIBLE.apply("first 12 then finally 33 apples") == "33"


def test_flexible_single_number():
    assert FLEXIBLE.apply("the total comes to 8") == "8"


def test_exact_match_ignores_commas_and_dollar():
    assert exact_match("$1,000", "1000", regexes_to_ignore=IGNORE, ignore_case=True) == 1


def test_exact_match_strips_trailing_period_and_gold_hash():
    assert exact_match("9.", "reasoning #### 9", regexes_to_ignore=IGNORE, ignore_case=True) == 1


def test_exact_match_mismatch():
    assert exact_match("8", "9", regexes_to_ignore=IGNORE, ignore_case=True) == 0


def test_exact_match_case_insensitive():
    assert exact_match("Ten", "ten", regexes_to_ignore=IGNORE, ignore_case=True) == 1
