
import pytest
from topicer.topic_discovery.time_text_detector import CzechTimeTextDetector


@pytest.fixture
def detector():
    det = CzechTimeTextDetector()
    return det


def test_czech_time_text_detector_no_entities(detector):
    text = "Text bez dat"
    assert not detector(text)


def test_czech_time_text_detector_with_entities(detector):
    text = "Karel IV. se narodil ve 14. století."

    assert detector(text)

    text = "Pyramidy byly postaveny 2500 př. n. l."

    assert detector(text)

