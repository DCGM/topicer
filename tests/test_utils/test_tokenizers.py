from pathlib import Path
from topicer.utils.tokenizers import CzechLemmatizedTokenizer, SpacyLemmatizedTokenizer

SCRIPT_DIR = Path(__file__).parent


def test_czech_lemmatized_tokenizer_plain():

    tokenizer = CzechLemmatizedTokenizer(
        tagger_file_path=SCRIPT_DIR / "fixtures" / "czech-morfflex2.0-pdtc1.0-220710.tagger.large_fixture",
    )

    text = "Toto je testovací text číslo 123."
    tokens = tokenizer.tokenize(text)

    expected_tokens = ["tento", "být", "testovací", "text", "číslo"]
    assert tokens == expected_tokens, f"Expected {expected_tokens}, but got {tokens}"


def test_czech_lemmatized_tokenizer_with_stopwords():
    tokenizer = CzechLemmatizedTokenizer(
        stopwords=["být", "je"],
        tagger_file_path=SCRIPT_DIR / "fixtures" / "czech-morfflex2.0-pdtc1.0-220710.tagger.large_fixture",
    )

    text = "Toto je testovací text číslo 123."
    tokens = tokenizer.tokenize(text)

    expected_tokens = ["tento", "testovací", "text", "číslo"]
    assert tokens == expected_tokens, f"Expected {expected_tokens}, but got {tokens}"


def test_spacy_lemmatized_tokenizer_plain():
    tokenizer = SpacyLemmatizedTokenizer()
    text = "This is a testing text number 123."

    tokens = tokenizer.tokenize(text)
    expected_tokens = ["this", "testing", "text", "number"]
    assert tokens == expected_tokens, f"Expected {expected_tokens}, but got {tokens}"


def test_spacy_lemmatized_tokenizer_with_stopwords():
    tokenizer = SpacyLemmatizedTokenizer(stopwords=["be", "this"])
    text = "This is a testing text number 123."

    tokens = tokenizer.tokenize(text)
    expected_tokens = ["testing", "text", "number"]
    assert tokens == expected_tokens, f"Expected {expected_tokens}, but got {tokens}"



