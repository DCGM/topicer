from pathlib import Path
import logging

import torch
from classconfig import ConfigurableMixin, ConfigurableValue
from transformers import AutoTokenizer, AutoModel

from topicer.base import BaseTopicer, MissingServiceError
from topicer.schemas import TagSpanProposal, TextChunk, DBRequest, Tag, TextChunkWithTagSpanProposals


logger = logging.getLogger(__name__)


class FailedToLoadModelError(Exception):
    pass


def cross_dot_product(
    model_outputs: torch.Tensor,
    sequence_ids: torch.Tensor,
    normalize_score: bool = True,
) -> torch.Tensor:
    """
    Calculates cross dot product between topic and text tokens based on their embeddings and sequence ids.

    Parameters:
        model_outputs (torch.Tensor): The output embeddings from the model, shape (seq_len, hidden_dim).
        sequence_ids (torch.Tensor): 0 for topic tokens, 1 for text tokens, -1 for special tokens (e.g.
            [CLS]/[SEP] or <bos>/<eos>, depending on the tokenizer). Using sequence ids rather than
            token_type_ids matters: token_type_ids is a BERT-specific convention and some tokenizers
            (e.g. ModernBERT-style ones like mmBERT) don't populate it meaningfully for pair inputs
            (it comes back all zeros), while sequence ids come from the tokenizer's actual pair-encoding
            logic and work correctly regardless of model family. Special tokens are excluded automatically
            since they're neither 0 nor 1 -- no manual slicing needed.
        normalize_score (bool, default=True): Whether to normalize the similarity scores by the square root of the embedding dimension.

    Returns:
        torch.Tensor: The similarity matrix between topic and text tokens.
    """
    topic_mask = (sequence_ids == 0)
    text_mask = (sequence_ids == 1)

    topic_tokens = model_outputs[topic_mask]
    text_tokens = model_outputs[text_mask]

    similarity_matrix = torch.matmul(topic_tokens, text_tokens.T)  # shape (topic_len, text_len)
    if normalize_score:
        similarity_matrix = similarity_matrix / torch.sqrt(torch.tensor(model_outputs.shape[-1], dtype=torch.float32))

    return similarity_matrix


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/touching (start, end) character spans, e.g. from
    predictions on overlapping tokenizer windows."""
    if not spans:
        return []

    spans = sorted(spans)
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:  # overlapping or touching
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


class CrossBertTopicer(BaseTopicer, ConfigurableMixin):
    model: str = ConfigurableValue(desc="Either a HuggingFace model name or a local path to the model directory.", user_default="UWB-AIR/Czert-B-base-cased")
    threshold: float = ConfigurableValue(desc="Threshold for topic tagging", user_default=0.5)
    device: str = ConfigurableValue(desc="Device to run the model on, either 'cuda' or 'cpu'.", user_default="cuda")
    max_length: int = ConfigurableValue(desc="Maximum sequence length for tokenization.", user_default=512)
    stride: int = ConfigurableValue(desc="Stride, in tokens, between overlapping windows when the tag + text exceeds max_length.", user_default=32)
    gap_tolerance: int = ConfigurableValue(desc="Tolerance for gaps of tokens between two spans of the same tag.", user_default=0)
    normalize_score: bool = ConfigurableValue(desc="Whether to normalize the scores. This is dependent on the specific model implementation.", user_default=True)
    soft_max_score: bool = ConfigurableValue(desc="Whether to use soft maximum when selecting score for each token.", user_default=True)
    loaded_from_huggingface: bool = False

    def __post_init__(self) -> None:
        self.load_model()
        self._model.to(self.device)
        self._model.eval()

    def load_model_from_hf(self) -> None:
        """
        Loads a CrossBertTopicer model from HuggingFace.
        """
        logger.debug(f"Loading CrossBertTopicer model from HuggingFace: {self.model} ...")
        self._model = AutoModel.from_pretrained(self.model)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.loaded_from_huggingface = True

    def load_local_model(self) -> None:
        """
        Loads a traced CrossBertTopicer model from a local path.
        """
        logger.debug(f"Loading CrossBertTopicer model from local path: {self.model} ...")
        model_path = Path(self.model) / f"model_{self.device}.pt"
        self._model = torch.jit.load(str(model_path), map_location=self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)

    def load_model(self):
        """
        Attempts to load model based on the provided configuration. It first tries to load from HuggingFace, and if that fails, it attempts to load from a local path.

        Raises:
            FailedToLoadModelError: If the model cannot be loaded from either source.
        """
        try:
            self.load_model_from_hf()
            return
        except (OSError, ValueError):
            pass

        try:
            self.load_local_model()
            return
        except (OSError, ValueError) as e:
            raise FailedToLoadModelError(f"Failed to load CrossBertTopicer model from both HuggingFace and local path: {self.model}\n{e}")

    def check_init(self):
        pass

    def prepare_chunks(self, chunk_text: str, tag_text: str) -> list[dict]:
        """
        Tokenizes the combined tag and text chunk, splitting the text into overlapping windows of at
        most max_length tokens when needed (keeping the tag tokens whole in every window). This mirrors
        the windowing done at training time (token_topicer/utils.py:prepare_sample_for_model), which
        matters because a plain truncation at max_length would silently drop text beyond that point
        instead of covering it via multiple windows the way the model saw at training time.

        Parameters:
            chunk_text (str): The text chunk to be analyzed.
            tag_text (str): The tag text (name [+ description]).

        Returns:
            list[dict]: One dict per window, each with input_ids, attention_mask, sequence_ids
                (0=tag, 1=text, None=special token) and offset_mapping.
        """
        tokenizer_output = self._tokenizer(
            tag_text,
            chunk_text,
            return_tensors="pt",
            truncation=False,
            return_offsets_mapping=True,
            return_attention_mask=True,
        )
        input_ids = tokenizer_output["input_ids"].squeeze(0)
        attention_mask = tokenizer_output["attention_mask"].squeeze(0)
        offsets = tokenizer_output["offset_mapping"].squeeze(0).tolist()
        sequence_ids = tokenizer_output.sequence_ids(0)

        if len(input_ids) <= self.max_length:
            return [{
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "sequence_ids": sequence_ids,
                "offset_mapping": offsets,
            }]

        # Too long -- split only the *text* tokens into overlapping windows, keeping the tag
        # (and special tokens) whole in every window. Assumes the tokenizer's pair layout puts
        # all non-text tokens as a contiguous prefix + contiguous suffix around the text tokens
        # (true for standard BERT/RoBERTa/ModernBERT-style pair encodings).
        text_positions = [i for i, s in enumerate(sequence_ids) if s == 1]
        if not text_positions:
            logger.warning("Tag (+ special tokens) alone exceeds max_length; skipping tag=%r", tag_text[:80])
            return []

        prefix_positions = list(range(0, text_positions[0]))
        suffix_positions = list(range(text_positions[-1] + 1, len(sequence_ids)))
        non_text_len = len(prefix_positions) + len(suffix_positions)

        budget = self.max_length - non_text_len
        if budget <= 0:
            logger.warning(
                "Tag (+ special tokens) is %d tokens, >= max_length=%d; skipping tag=%r",
                non_text_len, self.max_length, tag_text[:80],
            )
            return []

        stride = self.stride
        if stride >= budget:
            # Guard against a non-advancing loop if stride is too large for the budget.
            stride = max(budget // 2, 0)

        chunks = []
        n_text_tokens = len(text_positions)
        start = 0
        while True:
            end = min(start + budget, n_text_tokens)
            window_positions = text_positions[start:end]
            positions = prefix_positions + window_positions + suffix_positions
            idx = torch.tensor(positions, dtype=torch.long)

            chunks.append({
                "input_ids": input_ids[idx],
                "attention_mask": attention_mask[idx],
                "sequence_ids": [sequence_ids[p] for p in positions],
                "offset_mapping": [offsets[p] for p in positions],
            })

            if end == n_text_tokens:
                break
            start = end - stride

        return chunks

    def calculate_probabilities(self, model_outputs: torch.Tensor, sequence_ids: torch.Tensor) -> torch.Tensor:
        """
        Calculates tag probabilities for each text token in a single window based on model outputs and sequence ids.

        Parameters:
            model_outputs (torch.Tensor): The output embeddings from the model, shape (seq_len, hidden_dim).
            sequence_ids (torch.Tensor): 0=tag token, 1=text token, -1=special token.

        Returns:
            torch.Tensor: The probabilities of the tag for each text token in the window.
        """
        similarity_matrix = cross_dot_product(
            model_outputs=model_outputs,
            sequence_ids=sequence_ids,
            normalize_score=self.normalize_score,
        )
        max_similarities = torch.max(similarity_matrix, dim=0)[0] if not self.soft_max_score else torch.logsumexp(similarity_matrix, dim=0)
        max_similarities = torch.sigmoid(max_similarities)
        return max_similarities

    def run_chunk(self, chunk: dict) -> torch.Tensor:
        """
        Runs the model on a single tokenized window and returns per-text-token tag probabilities.

        Parameters:
            chunk (dict): One window as produced by prepare_chunks.

        Returns:
            torch.Tensor: The probabilities of the tag for each text token in the window.
        """
        input_ids = chunk["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = chunk["attention_mask"].unsqueeze(0).to(self.device)
        sequence_ids = torch.tensor(
            [-1 if s is None else s for s in chunk["sequence_ids"]],
            dtype=torch.long,
        ).to(self.device)

        model_outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
        if self.loaded_from_huggingface:
            try:
                model_outputs = model_outputs.last_hidden_state
            except AttributeError:
                model_outputs = model_outputs
        model_outputs = model_outputs.squeeze(0)

        return self.calculate_probabilities(model_outputs, sequence_ids)

    def propose_tag(self, text_chunk: TextChunk, tag: Tag) -> list[TagSpanProposal]:
        """
        Proposes spans for a single tag in the given text chunk.

        Parameters:
            text_chunk (TextChunk): The text chunk to be analyzed.
            tag (Tag): The tag to be proposed.

        Returns:
            list[TagSpanProposal]: A list of proposed tag spans.
        """
        chunk_text = text_chunk.text
        tag_text = tag.name + (f" - {tag.description}" if tag.description is not None else "")

        chunks = self.prepare_chunks(chunk_text, tag_text)

        all_spans: list[tuple[int, int]] = []
        for chunk in chunks:
            tag_probabilities = self.run_chunk(chunk)
            predictions = (tag_probabilities >= self.threshold).long().cpu().tolist()
            all_spans.extend(self.extract_char_spans(predictions, chunk["offset_mapping"]))

        # Overlapping windows can each independently (re)detect the same or a
        # partially overlapping span -- merge before returning.
        merged_spans = merge_spans(all_spans)

        return [
            TagSpanProposal(tag=tag, span_start=start, span_end=end, confidence=None, reason=None)
            for start, end in merged_spans
        ]

    def extract_char_spans(self, predictions: list[int], offset_mapping: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """
        Finds character (start, end) spans of text that correspond to positive model predictions within
        a single window. Accounts for gaps in model predictions based on gap_tolerance.

        Parameters:
            predictions (list[int]): The list of binary predictions for each text token in the window.
            offset_mapping (list[tuple[int, int]]): The list of character offsets for each token in the window.

        Returns:
            list[tuple[int, int]]: A list of (start_char, end_char) spans.
        """
        gap_tolerance = self.gap_tolerance

        char_spans = []
        start_char, end_char = None, None
        gap_count = 0

        for pred, (offset_start, offset_end) in zip(predictions, offset_mapping[-len(predictions)-1:-1], strict=True):
            if pred == 1:
                if start_char is None:
                    start_char = offset_start
                    end_char = offset_end
                    gap_count = 0
                else:
                    if gap_count > 0:
                        end_char = offset_end
                        gap_count = 0
                    else:
                        end_char = offset_end
            else:
                if start_char is not None:
                    gap_count += 1
                    if gap_count > gap_tolerance:
                        char_spans.append((start_char, end_char))
                        start_char, end_char = None, None
                        gap_count = 0

        if start_char is not None:
            char_spans.append((start_char, end_char))

        return char_spans

    async def propose_tags(self, text_chunk: TextChunk, tags: list[Tag]) -> TextChunkWithTagSpanProposals:
        """
        Finds presence and locations of given tags in the provided text chunk.

        Parameters:
            text_chunk (TextChunk): The text chunk to be analyzed.
            tags (list[Tag]): The list of tags to be proposed.

        Returns:
            TextChunkWithTagSpanProposals: The text chunk with proposed tag spans.
        """
        proposals = []
        for tag in tags:
            proposals.extend(self.propose_tag(text_chunk, tag))

        result = TextChunkWithTagSpanProposals(
            id=text_chunk.id,
            text=text_chunk.text,
            tag_span_proposals=proposals,
        )
        return result

    async def propose_tags_in_db(self, tag: Tag, db_request: DBRequest) -> list[TextChunkWithTagSpanProposals]:
        """
        Finds a presence and location of a given tag for text chunks retrieved from the database based on the provided DB request.

        Parameters:
            tag (Tag): The tag to be proposed.
            db_request (DBRequest): The database request to retrieve text chunks.

        Returns:
            list[TextChunkWithTagSpanProposals]: A list of text chunks with proposed tag spans.

        Raises:
            MissingServiceError: If the database connection is not set.
        """
        if self.db_connection is None:
            raise MissingServiceError("DB connection is not set for CrossBertTopicer. This can happen if the class is not properly initialized.")

        text_chunks = self.db_connection.get_text_chunks(db_request)
        return [self.propose_tags(text_chunk, [tag]) for text_chunk in text_chunks]
