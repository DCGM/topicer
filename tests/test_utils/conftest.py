import pytest
from topicer.utils.fuzzy_matcher import FuzzyMatcher


@pytest.fixture
def matcher():
    return FuzzyMatcher(max_dist_ratio=0.2)
