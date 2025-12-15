from typing import Sequence
from pathlib import Path
import logging

import torch
from classconfig import ConfigurableMixin, ConfigurableValue
from transformers import AutoTokenizer, AutoModel

from topicer.base import BaseTopicer, MissingServiceError
from topicer.schemas import DiscoveredTopicsSparse, TagSpanProposal, TextChunk, DiscoveredTopics, DBRequest, Tag, TextChunkWithTagSpanProposals, TagSpanProposal


logger = logging.getLogger(__name__)


class FailedToLoadModelError(Exception):
    pass


class CrossBertTopicer(BaseTopicer, ConfigurableMixin):
    model: str = ConfigurableValue(desc="Either a HuggingFace model name or a local path to the model directory.", user_default="UWB-AIR/Czert-B-base-cased")
    threshold: float = ConfigurableValue(desc="Threshold for topic tagging", user_default=0.5)
    device: str = ConfigurableValue(desc="Device to run the model on, either 'cuda' or 'cpu'.", user_default="cuda")
    max_length: int = ConfigurableValue(desc="Maximum sequence length for tokenization.", user_default=512)
    gap_tolerance: int = ConfigurableValue(desc="Tolerance for gaps of tokens between two spans of the same tag.", user_default=0)
    normalize_score: bool = ConfigurableValue(desc="Whether to normalize the scores. This is dependent on the specific model implementation.", user_default=True)
    soft_max_score: bool = ConfigurableValue(desc="Whether to use soft maximum when selecting score for each token.", user_default=False)
    loaded_from_huggingface: bool = False

    def __post_init__(self):
        self.load_model()
        self._model.to(self.device)
        self._model.eval()

    def load_model_from_hf(self):
        logger.debug(f"Loading CrossBertTopicer model from HuggingFace: {self.model} ...")
        self._model = AutoModel.from_pretrained(self.model)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.loaded_from_huggingface = True

    def load_local_model(self):
        logger.debug(f"Loading CrossBertTopicer model from local path: {self.model} ...")
        model_path = Path(self.model) / f"model_{self.device}.pt"
        self._model = torch.jit.load(str(model_path), map_location=self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)

    def load_model(self):
        try:
            self.load_model_from_hf()
            return
        except (OSError, ValueError):
            pass

        try:
            self.load_local_model()
            return
        except (OSError, ValueError):
            raise FailedToLoadModelError(f"Failed to load CrossBertTopicer model from both HuggingFace and local path: {self.model}")

    def check_init(self):
        pass

    async def discover_topics_sparse(self, texts: Sequence[TextChunk], n: int | None = None) -> DiscoveredTopicsSparse:
        raise NotImplementedError("Sparse topic discovery is not supported by CrossBertTopicer.")

    async def discover_topics_dense(self, texts: Sequence[TextChunk], n: int | None = None) -> DiscoveredTopics:
        raise NotImplementedError("Dense topic discovery is not supported by CrossBertTopicer.")
    
    async def discover_topics_in_db_sparse(self, db_request: DBRequest, n: int | None = None) -> DiscoveredTopicsSparse:
        raise NotImplementedError("Sparse topic discovery from DB is not supported by CrossBertTopicer.")

    async def discover_topics_in_db_dense(self, db_request: DBRequest, n: int | None = None) -> DiscoveredTopics:
        raise NotImplementedError("Dense topic discovery from DB is not supported by CrossBertTopicer.")
    
    def tokenize(self, chunk_text: str, tag_text: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenizer_output = self._tokenizer(
            tag_text,
            chunk_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        input_ids = tokenizer_output["input_ids"].to(self.device)
        attention_mask = tokenizer_output["attention_mask"].to(self.device)
        token_type_ids = tokenizer_output["token_type_ids"].to(self.device)
        offset_mapping = tokenizer_output["offset_mapping"].squeeze(0).tolist()
        return input_ids, attention_mask, token_type_ids, offset_mapping
    
    def calculate_probabilities(self, model_outputs: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        similarity_matrix = self.cross_dot_product(
            model_outputs=model_outputs,
            token_type_ids=token_type_ids,
        )
        max_similarities, _ = torch.max(similarity_matrix, dim=0) if not self.soft_max_score else torch.logsumexp(similarity_matrix, dim=0)
        max_similarities = torch.sigmoid(max_similarities)
        return max_similarities

    def propose_tag(self, text_chunk: TextChunk, tag: Tag) -> list[TagSpanProposal]:
        chunk_text = text_chunk.text
        tag_text = tag.name + (f" - {tag.description}" if tag.description is not None else "")
    
        input_ids, attention_mask, token_type_ids, offset_mapping = self.tokenize(chunk_text, tag_text)
        model_outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
        if self.loaded_from_huggingface:
            model_outputs = model_outputs.last_hidden_state
        tag_probabilities = self.calculate_probabilities(model_outputs, token_type_ids)
        predictions = (tag_probabilities >= self.threshold).long().cpu().tolist()

        spans = self.extract_spans(predictions, offset_mapping, tag)
        return spans

    def extract_spans(self, predictions: list[int], offset_mapping: list[tuple[int, int]], tag: Tag) -> list[TagSpanProposal]:
        gap_tolerance = self.gap_tolerance

        char_spans = []
        start_char, end_char = None, None
        gap_count = 0

        for pred, (offset_start, offset_end) in zip(predictions, offset_mapping):
            if pred == 1:
                if start_char is None:
                    start_char = offset_start
                    end_char = offset_end
                    gap_count = 0
                else:
                    if gap_count > 0:
                        # Extend the end to include the gap
                        end_char = offset_end
                        gap_count = 0
                    else:
                        end_char = offset_end
            else:
                if start_char is not None:
                    gap_count += 1
                    if gap_count > gap_tolerance:
                        # Close the span
                        char_spans.append(TagSpanProposal(
                            tag=tag,
                            span_start=start_char,
                            span_end=end_char,
                            confidence=1.0,  # Placeholder confidence
                            reason=None
                        ))
                        start_char, end_char = None, None
                        gap_count = 0

        if start_char is not None:
            char_spans.append(TagSpanProposal(
                tag=tag,
                span_start=start_char,
                span_end=end_char,
                confidence=1.0,  # Placeholder confidence
                reason=None
            ))

        return char_spans

    async def propose_tags(self, text_chunk: TextChunk, tags: list[Tag]) -> TextChunkWithTagSpanProposals:
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
        if self.db_connection is None:
            raise MissingServiceError("DB connection is not set for CrossBertTopicer. This can happen if the class is not properly initialized.")
        
        text_chunks = self.db_connection.get_text_chunks(db_request)
        return [self.propose_tags(text_chunk, [tag]) for text_chunk in text_chunks]

    def cross_dot_product(
        self,
        model_outputs: torch.Tensor,
        token_type_ids: torch.Tensor,
    ):
        topic_mask = (token_type_ids == 0)
        text_mask = (token_type_ids == 1)

        topic_tokens = model_outputs[topic_mask.bool()][1:-1]  # exclude CLS and SEP
        text_tokens = model_outputs[text_mask.bool()][:-1]     # exclude SEP

        similarity_matrix = torch.matmul(topic_tokens, text_tokens.T)  # shape (topic_len, text_len)
        if self.normalize_score:
            similarity_matrix = similarity_matrix / torch.sqrt(torch.tensor(model_outputs.shape[-1], dtype=torch.float32))

        return similarity_matrix
