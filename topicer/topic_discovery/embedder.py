import logging

import numpy as np
import torch
from classconfig import ConfigurableValue, RelativePathTransformer, ConfigurableMixin, CreatableMixin
from numpy._typing import NDArray
from ruamel.yaml.scalarstring import LiteralScalarString
from sentence_transformers import SentenceTransformer


class DocEmbedder(ConfigurableMixin, CreatableMixin):
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

    def __post_init__(self):
        kwargs = {}
        if "bge-multilingual-gemma2" in self.model:
            logging.warning(
                "HOTFIX: Casting model to FP16, as this is essential for gemma2. See https://huggingface.co/BAAI/bge-multilingual-gemma2/discussions/2")
            kwargs = {"torch_dtype": torch.float16}
        self.transformer = SentenceTransformer(
            self.model, cache_folder=self.cache_folder, model_kwargs=kwargs, device=self.device
        )

    def encode(self, docs: list[str], show_progress_bar: bool = False, normalize_embeddings: bool | None = None) -> NDArray:
        """
        Encodes given documents into embeddings.

        :param docs: list of documents to encode
        :param show_progress_bar: whether to show a progress bar during encoding
        :param normalize_embeddings: whether to normalize the embeddings using L2 normalization
            if not provided, uses the default setting
        :return: list of embeddings
        """
        embeddings = self.transformer.encode(
            docs,
            prompt=self.prompt,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.normalize_embeddings if normalize_embeddings is None else normalize_embeddings,
            batch_size=self.batch_size
        )

        if self.return_fp32:
            embeddings = embeddings.astype(np.float32)

        return embeddings
