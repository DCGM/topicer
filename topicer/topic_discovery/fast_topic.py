import asyncio
import copy
import uuid
import json_repair

from typing import Sequence

import numpy as np
from classconfig import ConfigurableMixin, CreatableMixin, ConfigurableValue, ConfigurableFactory, \
    ConfigurableSubclassFactory
from numpy._typing import NDArray
from pydantic import BaseModel, Field
from ruamel.yaml.scalarstring import LiteralScalarString
from topmost.preprocess.preprocess import Tokenizer, Preprocess

from topicer.llm_api import APIAsync, OpenAsyncAPI, APIRequest
from topicer.schemas import DBRequest, DiscoveredTopics, TextChunk, DiscoveredTopicsSparse, Topic
from topicer.topic_discovery import TopicDiscovery
from topicer.embedding.local import LocalEmbedder
from fastopic import FASTopic

from topicer.utils.template import TemplateTransformer, Template
from topicer.utils.tokenizers import CzechLemmatizedTokenizer


class GenerateTopicNameResponse(BaseModel):
    """
    Schema for the response from the topic name generation LLM.
    """
    explanation: str = Field(..., description="Explanation of the topic name.")
    name: str = Field(..., description="Name given to the topic.")


class FastTopicDiscovery(TopicDiscovery, ConfigurableMixin, CreatableMixin):
    """
    Performs topic discovery using FastTopic library (https://github.com/bobxwu/FASTopic).
    """
    n_topics: int = ConfigurableValue(desc="Number of topics to discover.", user_default=10)
    api: APIAsync = ConfigurableSubclassFactory(
        APIAsync,
        desc="Asynchronous LLM API used for obtaining topic names and descriptions.",
        user_default=OpenAsyncAPI,
    )
    embedder: LocalEmbedder = ConfigurableFactory(
        LocalEmbedder,
        desc="Document embedder used to convert texts into embeddings for topic discovery.",
    )
    tokenizer: Tokenizer = ConfigurableSubclassFactory(
        Tokenizer,
        desc="Tokenizer used to preprocess texts before topic discovery.",
        user_default=CzechLemmatizedTokenizer
    )
    vocab_size: int = ConfigurableValue(
        desc="Size of the vocabulary for topic modeling.",
        user_default=5000,
    )
    verbose: bool = ConfigurableValue(
        desc="Whether to print verbose output during topic discovery.",
        user_default=False,
    )
    topic_rep_size: int = ConfigurableValue(
        desc="Number of top words to represent each topic.",
        user_default=15,
    )
    topic_doc_rep_size: int = ConfigurableValue(
        desc="Number of top documents to represent each topic.",
        user_default=3,
    )
    generate_topic_name_model: str = ConfigurableValue(
        desc="Model to use for generating topic names.",
        user_default="gpt-5-mini",
    )
    generate_topic_name_system_prompt: Template = ConfigurableValue(
        desc="System prompt for generating topic names.",
        user_default=LiteralScalarString("""Pojmenuj skupiny slov na základě jejich společného tématu a pro kontext využij přiložené dokumenty.

Příklad:
Slova: jablko, banán, hrozen, slunečnice
Dokumenty:
Jablka se pěstují v mnoha částech světa a jsou bohatá na vitamíny.
Banány jsou tropické ovoce, které je bohaté na draslík.
Slunečnice jsou nádherné rostliny, které rostou vysoko a následují slunce.

{
    "explanation": "\"Jablko\", \"banán\" a \"hrozen\" patří mezi ovoce. \"Slunečnice\" je rostlina. Dokument 1 je o jablkách. Dokument 2 je o banánech. Dokument 3 je o slunečnicích. Takže název tématu by mohl být:",
    "name": "Ovoce a rostliny"
}

Vrať názvy ve formátu JSON:

{
    "explanation": "Vysvětlení",
    "name": "Název",
}

Důležité: Výstupem musí být pouze tento JSON, bez dalšího textu.
Důležité: Snaž se téma popsat jedním výrazem (např. "ovoce" nebo "rostliny"). Nepoužívej spojku "a", pokud to není nezbytně nutné.
Důležité: Ujisti se, že ve vysvětlení zmíníš každý dokument (očísluj je jako 1., 2., 3.!).
Důležité: Nezapomeň nejprve rozebrat roli slov a až poté dokumenty.
"""),
        transform=TemplateTransformer()
    )
    generate_topic_name_prompt: Template = ConfigurableValue(
        desc="Jinja2 template prompt for generating topic names.",
        user_default=LiteralScalarString("""Analyzuj následující:
Slova: {{ topic_words | join(", ") }}
Dokumenty:
{% for doc in topic_docs %}
{{doc.text}}
{% endfor %}

- Vysvětli, proč každé slovo a dokument zapadá do tématu (nebo nezapadá).
- Navrhni název pro tuto skupinu.
"""),
        transform=TemplateTransformer()
    )

    sparse_threshold: float = ConfigurableValue(
        desc="Threshold for sparsity in topic-document distributions. Can be combined with sparse_top_k.",
        user_default=0.01,
        validator=lambda value: 0.0 <= value <= 1.0
    )

    sparse_top_k: int | None = ConfigurableValue(
        desc="Number of top topics to keep per document in sparse representation. Can be combined with sparse_threshold.",
        user_default=10,
        validator=lambda value: value is None or value > 0
    )

    max_char_length: int = ConfigurableValue(
        desc="Maximum character length of texts. Longer texts will be truncated from the end before processing.",
        user_default=1024,
        validator=lambda value: value > 0,
        voluntary=True
    )

    async def discover_topics(self, texts: Sequence[TextChunk], n: int | None = None,
                              sparse: bool = True) -> DiscoveredTopics | DiscoveredTopicsSparse:
        texts = copy.deepcopy(texts)
        self.truncate_texts(texts, self.max_char_length)

        model = await asyncio.to_thread(self._create_fastopic_model, n=n)

        top_words, doc_topic_dist = await asyncio.to_thread(
            model.fit_transform,
            docs=[text.text for text in texts]
        )

        return await self._process_topics(
            top_words=top_words,
            doc_topic_dist=doc_topic_dist,
            texts=texts,
            sparse=sparse
        )

    async def discover_topics_in_db(self, db_request: DBRequest, n: int | None = None,
                                    sparse: bool = True) -> DiscoveredTopics | DiscoveredTopicsSparse:
        pass

    @staticmethod
    def truncate_texts(texts: Sequence[TextChunk], max_char_length: int):
        """
        Truncates texts to the specified maximum character length.
        Works in-place.

        :param texts: Sequence of TextChunk objects.
        :param max_char_length: Maximum character length for each text.
        """
        for text_chunk in texts:
            if len(text_chunk.text) > max_char_length:
                text_chunk.text = text_chunk.text[:max_char_length]

    def _create_fastopic_model(self, n: int | None = None) -> FASTopic:
        """
        Creates and configures the FASTopic model.

        :param n: Optional number of topics to discover. If None, uses the default from configuration.
        :return: Configured FASTopic model.
        """
        preprocessing = Preprocess(tokenizer=self.tokenizer, vocab_size=self.vocab_size)
        model = FASTopic(
            num_topics=self.n_topics if n is None else n,
            preprocess=preprocessing,
            num_top_words=self.topic_rep_size,
            doc_embed_model=self.embedder,
            verbose=self.verbose,
            normalize_embeddings=self.embedder.normalize_embeddings # Fast topic passes this to encode method
        )
        return model

    @staticmethod
    def get_top_k_docs_per_topic(topic_doc_dist: NDArray, texts: Sequence[TextChunk], k: int) -> list[list[TextChunk]]:
        """
        Retrieves the top K documents for each topic based on the topic-document distribution.

        :param topic_doc_dist: document distributions of topics (topic-doc distributions), a numpy array with shape A × B (number of topics A and number of documents B ).
        :param texts: Original text chunks.
        :param k: Number of top documents to retrieve per topic.
        :return: List of lists containing top k TextChunks for each topic.
        """

        num_topics = topic_doc_dist.shape[0]
        top_docs_per_topic = []

        for topic_idx in range(num_topics):
            topic_distribution = topic_doc_dist[topic_idx]
            top_doc_indices = topic_distribution.argsort()[-k:][::-1]
            top_docs = [texts[i] for i in top_doc_indices]
            top_docs_per_topic.append(top_docs)

        return top_docs_per_topic

    @staticmethod
    def sparsify_topic_document_distribution(topic_doc_dist: NDArray, threshold: float, k: int | None) -> list[list[tuple[int, float]]]:
        """
        Converts the dense topic-document distribution into a sparse representation based on the configured threshold and top-k.

        :param topic_doc_dist: Topic-document distribution matrix (A × B) where A is the number of topics and B is the number of documents.
        :param threshold: Minimum weight threshold to include a document in the sparse representation.
        :param k: Optional number of top documents to keep per topic.
        :return: Sparse representation of the topic-document distribution.
            documents are always sorted by index within each topic.
        """
        n_topics, n_docs = topic_doc_dist.shape

        if k is not None and k < n_docs:
            top_k_idx = np.argpartition(topic_doc_dist, -k, axis=1)[:, -k:]

            final_indices = np.sort(top_k_idx, axis=1)
            final_weights = np.take_along_axis(topic_doc_dist, final_indices, axis=1)
        else:
            final_indices = np.tile(np.arange(n_docs), (n_topics, 1))
            final_weights = topic_doc_dist

        mask = final_weights >= threshold

        sparse_representation = []

        for indices, weights, row_mask in zip(final_indices, final_weights, mask):
            # We only keep elements where the mask is True
            valid_indices = indices[row_mask]
            valid_weights = weights[row_mask]

            sparse_representation.append([
                (int(i), float(w)) for i, w in zip(valid_indices, valid_weights)
            ])

        return sparse_representation

    async def _generate_topic_names(self, top_words: list[list[str]], top_docs_per_topic: list[list[TextChunk]]) -> list[tuple[str, str]]:
        """
        Generates names for each topic using the LLM API.

        :param top_words: List of top words for each topic.
        :param top_docs_per_topic: List of lists containing top documents for each topic.
        :return: List of tuples:
            generated topic name
            explanation of the topic name
        """

        topic_names = [None] * len(top_words)  # Pre-allocate list to maintain order

        async def process_and_store(index: int, words: list[str], docs: list[TextChunk]):
            prompt = self.generate_topic_name_prompt.render({
                "topic_words": words,
                "topic_docs": docs
            })

            request = APIRequest(
                custom_id=f"generate_topic_name_{uuid.uuid4()}",
                model=self.generate_topic_name_model,
                messages=[
                    {"role": "system", "content": self.generate_topic_name_system_prompt.render({})},
                    {"role": "user", "content": prompt}
                ],
                response_format=GenerateTopicNameResponse,
            )

            # This await is now inside the task, allowing other tasks to run while this waits
            api_output = await self.api.process_single_request(request)
            content = api_output.response.get_raw_content()
            parsed = json_repair.loads(content)
            name = parsed["name"]
            explanation = parsed["explanation"]
            topic_names[index] = (name, explanation)

        async with asyncio.TaskGroup() as tg:
            for i, (words, docs) in enumerate(zip(top_words, top_docs_per_topic)):
                tg.create_task(process_and_store(i, words, docs))

        return topic_names

    async def _process_topics(self, top_words: list[str], doc_topic_dist: NDArray, texts: Sequence[TextChunk],
                              sparse: bool) -> DiscoveredTopics | DiscoveredTopicsSparse:
        """
        Processes the discovered topics and formats them into the appropriate schema.

        :param top_words: List of top words strings for each topic.
            The fast topic returns top words in single string separated by spaces.
        :param doc_topic_dist: topic distributions of documents (doc-topic distributions), a numpy array with shape N × K (number of documents N and number of topics K ).
        :param texts: Original text chunks.
        :param sparse: Whether to return sparse representation.
        """

        topic_doc_dist = doc_topic_dist.T
        top_docs_per_topic = await asyncio.to_thread(
            self.get_top_k_docs_per_topic, topic_doc_dist=topic_doc_dist, texts=texts, k=self.topic_doc_rep_size
        )
        top_words = [words.split() for words in top_words]
        topic_names_with_explanation = await self._generate_topic_names(top_words, top_docs_per_topic)

        topics = [
            Topic(name=name, name_explanation=explanation) for name, explanation in topic_names_with_explanation
        ]
        if sparse:
            topic_doc_dist = await asyncio.to_thread(
                self.sparsify_topic_document_distribution,
                topic_doc_dist=topic_doc_dist,
                threshold=self.sparse_threshold,
                k=self.sparse_top_k
            )
            return DiscoveredTopicsSparse(
                topics=topics,
                topic_documents=topic_doc_dist,
            )
        else:
            return DiscoveredTopics(
                topics=topics,
                topic_documents=topic_doc_dist.tolist(),
            )




