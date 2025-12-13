import logging

import numpy as np
import torch
from classconfig import ConfigurableValue, RelativePathTransformer, ConfigurableMixin, CreatableMixin
from ruamel.yaml.scalarstring import LiteralScalarString
from sentence_transformers import SentenceTransformer

from topicer.base import BaseEmbeddingService


class LocalEmbedder(BaseEmbeddingService, ConfigurableMixin, CreatableMixin):
    """
    Configurable embedder that runs locally.

    The embedder is based on sentence-transformers library.
    """
    model: str = ConfigurableValue(
        desc="Any model name from sentence-transformers library or compatible model from huggingface hub.",
        user_default="BAAI/bge-multilingual-gemma2"
    )
    cache_folder: str | None = ConfigurableValue(
        desc="Path to cache directory for the model.",
        voluntary=True,
        transform=RelativePathTransformer(allow_none=True),
        user_default=None
    )
    prompt: str | None = ConfigurableValue(
        desc="Prompt for the embedding model, if supported.",
        voluntary=True,
        user_default=LiteralScalarString("""<instruct>Represent this Czech historical document to find similar ones.
        <query>""")
    )
    query_prompt: str | None = ConfigurableValue(
        desc="Query prompt for the embedding model, used for query embedding methods.",
        voluntary=True,
        user_default=None
    )
    batch_size: int = ConfigurableValue(
        desc="Batch size for encoding documents.",
        user_default=32,
        validator=lambda x: x > 0
    )
    device: str | list[str] | None = ConfigurableValue(
        "Device(s) to use for computation. Can be: - A single device string (e.g., \"cuda:0\", \"cpu\") for single-process encoding - A list of device strings (e.g., [\"cuda:0\", \"cuda:1\"], [\"cpu\", \"cpu\", \"cpu\", \"cpu\"]) to distribute encoding across multiple processes - None to auto-detect available device for single-process encoding If a list is provided, multi-process encoding will be used. Defaults to None.",
        voluntary=True,
        user_default=None,
        validator=lambda x: x is None or isinstance(x, str) or (isinstance(x, list) and all(isinstance(dev, str) for dev in x))
    )
    return_fp32: bool = ConfigurableValue(
        desc="Whether to return embeddings as FP32 numpy arrays. If False, the embeddings will be in the default dtype of the model.",
        user_default=True,
        voluntary=True,
    )
    normalize_embeddings: bool = ConfigurableValue(
        desc="Whether to normalize the embeddings using L2 normalization.",
        user_default=False,
        voluntary=True,
    )
    show_progress_bar: bool = ConfigurableValue(
        desc="Whether to show a progress bar during embedding.",
        user_default=False,
        voluntary=True,
    )

    def __post_init__(self):
        kwargs = {}
        if "bge-multilingual-gemma2" in self.model:
            logging.warning(
                "HOTFIX: Casting model to FP16, as this is essential for gemma2. See https://huggingface.co/BAAI/bge-multilingual-gemma2/discussions/2")
            kwargs = {"torch_dtype": torch.float16}
        self.transformer = SentenceTransformer(
            self.model, cache_folder=self.cache_folder, model_kwargs=kwargs, device=self.device
        )

    def embed(self, text_chunks: list[str] | str, prompt: str | None = None, normalize: bool | None = None) -> np.ndarray:
        """
        Encodes given documents into embeddings.

        :param text_chunks: list of text chunks
        :param prompt: prompt for the embedding model
            If None it will use the default prompt from configuration.
        :param normalize: whether to normalize the embeddings using L2 normalization
        :return: embeddings in form of numpy array TEXT_CHUNKS X DIMENSION
        """
        if isinstance(text_chunks, str):
            text_chunks = [text_chunks]

        embeddings = self.transformer.encode(
            text_chunks,
            prompt=self.prompt if prompt is None else prompt,
            show_progress_bar=self.show_progress_bar,
            normalize_embeddings=self.normalize_embeddings if normalize is None else normalize,
            batch_size=self.batch_size
        )

        if self.return_fp32:
            embeddings = embeddings.astype(np.float32)

        return embeddings

    def embed_queries(self, queries: list[str], normalize: bool | None = None) -> np.ndarray:
        """
        Encodes given queries into embeddings using default query prompt if available.

        :param queries: list of queries to encode
        :param normalize: whether to normalize the embeddings using L2 normalization
        :return: embeddings in form of numpy array TEXT_CHUNKS X DIMENSION
        """
        return self.embed(queries, prompt=self.query_prompt, normalize=normalize)
