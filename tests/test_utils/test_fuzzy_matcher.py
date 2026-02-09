import pytest
from topicer.utils.fuzzy_matcher import FuzzyMatcher


@pytest.fixture
def matcher():
    return FuzzyMatcher(max_dist_ratio=0.2)


@pytest.mark.parametrize("input_text, expected", [
    ("  text   with  spaces  ", "text with spaces"),
    ("new\nline\ttabulator", "new line tabulator"),
    ("", ""),
    (None, ""),
])
def test_normalize_text(matcher, input_text, expected):
    assert matcher._normalize_text(input_text) == expected


def test_get_best_dist_exact(matcher):
    assert matcher._get_best_dist("hello", "here is hello world") == 0


def test_get_best_dist_fuzzy(matcher):
    # 'hello' vs 'hallo' -> distance 1
    assert matcher._get_best_dist("hello", "here is hallo world") == 1


def test_get_best_dist_too_different(matcher):
    # 'superman' vs 'hello' -> over max_dist_ratio (0.2 * 8 = 1.6 -> max dist 1)
    assert matcher._get_best_dist("superman", "hello") is None


def test_find_best_span_exact_match(matcher):
    full_text = "The quick brown fox jumps over the lazy dog."
    quote = "brown fox"
    # 'brown fox' starts at index 10 and covers characters up to (but not including) index 19, i.e. span [10, 19)
    assert matcher.find_best_span(full_text, quote) == (10, 19)


def test_find_best_span_with_typo_within_distance(matcher):
    full_text = "The quick brown panther jumps over the lazy dog."
    # 2 typos from "brown panther", but should be within max_dist_ratio (2/13 ~ 0.15 < 0.2)
    quote = "brawn pamther"
    # Length 13 * 0.2 = 2.6 -> allowed distance 2. Should pass.
    span = matcher.find_best_span(full_text, quote)
    assert span == (10, 23)


def test_find_best_span_with_typo_exceeding_distance(matcher):
    full_text = "The quick brown panther jumps over the lazy dog."
    # 3 typos from "brown panther", which exceeds max_dist_ratio (3/13 ~ 0.23 > 0.2)
    quote = "brawn pamter"
    # Length 13 * 0.2 = 2.6 -> allowed distance 2. Should fail and return None.
    span = matcher.find_best_span(full_text, quote)
    assert span is None


def test_find_best_span_no_match(matcher):
    full_text = "It is a sunny day and the birds are singing."
    quote = "rainy weather"
    assert matcher.find_best_span(full_text, quote) is None


def test_find_best_span_with_context(matcher):
    # We have two same quotes, but different contexts. We want to find the one with matching context.
    full_text = "An apple is red. A car is red and fast."
    quote = "red"

    # Without context, it would take the first occurrence
    assert matcher.find_best_span(full_text, quote) == (12, 15)

    # With context before ("A car is") it should find the second one
    span = matcher.find_best_span(full_text, quote, context_before="A car is")
    assert span == (26, 29)


def test_find_best_span_empty_inputs(matcher):
    assert matcher.find_best_span("", "something") is None
    assert matcher.find_best_span("something", "") is None


def test_context_penalty_fallback(matcher):
    # Test the situation where the context does not match at all (high penalty)
    full_text = "Hello world."
    quote = "world"
    # Context 'X Y Z' is not in the text, a fallback penalty should be calculated
    span = matcher.find_best_span(full_text, quote, context_before="X Y Z")
    assert span == (6, 11)  # It should still find at least the quote
