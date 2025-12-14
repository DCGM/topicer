import re
from abc import ABC, abstractmethod

from classconfig import ConfigurableValue, ConfigurableMixin


class TimeTextDetector(ABC):
    """
    Base class for detectors that can find time expressions in text.
    """

    @abstractmethod
    def __call__(self, text: str) -> bool:
        """
        Detects time expressions in the given text.

        :param text: text to detect time expressions in
        :return: whether time expressions were detected
        """
        ...


class CzechTimeTextDetector(TimeTextDetector, ConfigurableMixin):
    """
    Detector for Czech time expressions.
    """
    DATE_PATTERN = r"\b(?:\d{1,2}\.\s*(?:leden|únor|březen|duben|květen|červen|červenec|srpen|září|říjen|listopad|prosinec)|(?:leden|únor|březen|duben|květen|červen|červenec|srpen|září|říjen|listopad|prosinec)\s*\d{1,2}\.|(?:\d{1,2}\.\s?\d{1,2}\.\s?\d{3,4}))\b"
    CENTURY_PATTERN = r"\b(?:\d{1,2}\.\s*století|století\s*\d{1,2}\.)\b"
    ERA_PATTERN = r"\b(?:př\. n\. l\.|n\. l\.|př\. Kr\.|BC|AD)"
    YEAR_PATTERN = r"\b(?:1\d{3}|20\d{2})\b"
    ALL_PATTERNS = f"(?:{DATE_PATTERN}|{CENTURY_PATTERN}|{ERA_PATTERN}|{YEAR_PATTERN})"

    def __post_init__(self):
        self.search = re.compile(self.ALL_PATTERNS, re.IGNORECASE)

    def __call__(self, text: str) -> bool:
        return self.search.search(text) is not None
