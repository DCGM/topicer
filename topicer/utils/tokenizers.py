import spacy
from classconfig import ConfigurableValue, RelativePathTransformer
from topmost.preprocess.preprocess import Tokenizer, alpha, alpha_or_num

from corpy.morphodita import Tagger
from stop_words import get_stop_words


class CzechLemmatizedTokenizer(Tokenizer):
    stopwords: list[str] | str | None = ConfigurableValue(
        desc="List of stopwords to remove from the text during tokenization or language (available languages https://github.com/Alir3z4/stop-words).",
        validator=lambda x: isinstance(x, list) or isinstance(x, str) or x is None,
        user_default="czech"
    )
    keep_num: bool = ConfigurableValue(
        desc="Whether to keep numeric tokens in the text.",
        user_default=False
    )
    keep_alphanum: bool = ConfigurableValue(
        desc="Whether to keep alphanumeric tokens in the text.",
        user_default=False
    )
    strip_html: bool = ConfigurableValue(
        desc="Whether to strip HTML tags from the text before tokenization.",
        user_default=False
    )
    no_lower: bool = ConfigurableValue(
        desc="Whether to skip lowercasing of the text before tokenization.",
        user_default=False
    )
    min_length: int = ConfigurableValue(
        desc="Minimum length of tokens to keep.",
        user_default=3,
        validator=lambda x: isinstance(x, int) and x >= 0
    )
    tagger_file_path: str = ConfigurableValue(
        desc="Path to the Morphodita tagger file.",
        user_default="czech-morfflex2.0-pdtc1.0-220710.tagger",
        transform=RelativePathTransformer()
    )

    def __init__(self,
                 stopwords: list[str] | str | None = None,
                 keep_num: bool = False,
                 keep_alphanum: bool = False,
                 strip_html: bool = False,
                 no_lower: bool = False,
                 min_length: int = 3,
                 tagger_file_path: str = "czech-morfflex2.0-pdtc1.0-220710.tagger",
                 **kwargs):

        if stopwords is None:
            stopwords = []  # Default value for stopwords

        if isinstance(stopwords, str):
            stopwords = get_stop_words(stopwords)

            # Add the parameters to kwargs if they are expected by the superclass
        kwargs.update({
            'stopwords': stopwords,
            'keep_num': keep_num,
            'keep_alphanum': keep_alphanum,
            'strip_html': strip_html,
            'no_lower': no_lower,
            'min_length': min_length
        })

        # Initialize the Morphodita Tagger
        self.tagger = Tagger(tagger_file_path)

        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def tokenize(self, text):
        """
        Tokenize and lemmatize the input text using spaCy.
        This function only performs lemmatization and skips other NLP components.

        :param text: The input text to process.
        :return: A list of lemmatized tokens.
        """
        # Clean text
        text = self.clean_text(text, self.strip_html, self.lower)

        # Process the text using corpy pipeline
        tokens = [token.lemma for token in self.tagger.tag(text)]

        # Remove -[num] suffixes from tokens
        tokens = [token.split("-")[0] if not token.startswith("-") else token for token in tokens]
        tokens = [token.split("_")[0] if not token.startswith("_") else token for token in tokens]

        # Drop stopwords
        tokens = [token for token in tokens if token not in self.stopword_set]

        # Remove numeric tokens
        tokens = [token for token in tokens if not token.isdigit()]

        # # drop short tokens
        # if self.min_length > 0:
        #     tokens = [t if len(t) >= self.min_length else '_' for t in tokens]

        return tokens


class SpacyLemmatizedTokenizer(Tokenizer):
    stopwords: list[str] | str | None = ConfigurableValue(
        desc="List of stopwords to remove from the text during tokenization or language (available languages https://github.com/Alir3z4/stop-words).",
        validator=lambda x: isinstance(x, list) or isinstance(x, str) or x is None,
        user_default="czech"
    )
    keep_num: bool = ConfigurableValue(
        desc="Whether to keep numeric tokens in the text.",
        user_default=False
    )
    keep_alphanum: bool = ConfigurableValue(
        desc="Whether to keep alphanumeric tokens in the text.",
        user_default=False
    )
    strip_html: bool = ConfigurableValue(
        desc="Whether to strip HTML tags from the text before tokenization.",
        user_default=False
    )
    no_lower: bool = ConfigurableValue(
        desc="Whether to skip lowercasing of the text before tokenization.",
        user_default=False
    )
    min_length: int = ConfigurableValue(
        desc="Minimum length of tokens to keep.",
        user_default=3,
        validator=lambda x: isinstance(x, int) and x >= 0
    )

    def __init__(self,
                 stopwords: list[str] | str | None = None,
                 keep_num: bool = False,
                 keep_alphanum: bool = False,
                 strip_html: bool = False,
                 no_lower: bool = False,
                 min_length: int = 3,
                 **kwargs):
        if stopwords is None:
            stopwords = []  # Default value for stopwords

        if isinstance(stopwords, str):
            stopwords = get_stop_words(stopwords)

            # Add the parameters to kwargs if they are expected by the superclass
        kwargs.update({
            'stopwords': stopwords,
            'keep_num': keep_num,
            'keep_alphanum': keep_alphanum,
            'strip_html': strip_html,
            'no_lower': no_lower,
            'min_length': min_length
        })

        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def tokenize(self, text):
        """
        Tokenize and lemmatize the input text using spaCy.
        This function only performs lemmatization and skips other NLP components.

        :param text: The input text to process.
        :return: A list of lemmatized tokens.
        """
        # Clean text
        text = self.clean_text(text, self.strip_html, self.lower)

        # Process the text using spaCy's NLP pipeline
        doc = self.nlp(text)

        # Extract lemmatized tokens for all words
        tokens = [token.lemma_ for token in doc]

        # Drop stopwords
        tokens = [token for token in tokens if token not in self.stopword_set]

        # remove tokens that contain numbers
        if not self.keep_alphanum and not self.keep_num:
            tokens = [t for t in tokens if alpha.match(t)]

        # or just remove tokens that contain a combination of letters and numbers
        elif not self.keep_alphanum:
            tokens = [t for t in tokens if alpha_or_num.match(t)]

        # drop short tokens
        if self.min_length > 0:
            tokens = [t for t in tokens if len(t) >= self.min_length]

        return tokens
