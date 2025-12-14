import asyncio
import copy
import random
from typing import Sequence

import numpy as np
from classconfig import ConfigurableMixin, CreatableMixin, ConfigurableValue, ConfigurableSubclassFactory
from fastopic import FASTopic
from numpy._typing import NDArray
from pydantic import BaseModel, Field
from ruamel.yaml.scalarstring import LiteralScalarString
from topmost.preprocess.preprocess import Tokenizer, Preprocess

from topicer.base import BaseTopicer, MissingServiceError, BaseEmbeddingService
from topicer.schemas import DBRequest, DiscoveredTopics, TextChunk, DiscoveredTopicsSparse, Topic, Tag, \
    TextChunkWithTagSpanProposals
from topicer.topic_discovery.time_text_detector import TimeTextDetector, CzechTimeTextDetector
from topicer.utils.template import TemplateTransformer, Template
from topicer.utils.tokenizers import CzechLemmatizedTokenizer


class EmbeddingServiceWrapper:
    """
    Wrapper for embedding service to be used in FASTopic.
    """
    def __init__(self, embedding_service: BaseEmbeddingService):
        self.embedding_service = embedding_service

    def encode(self, docs: list[str], normalize_embeddings: bool | None = None, show_progress_bar: bool = False) -> NDArray:
        """
        Encodes given documents into embeddings.

        :param docs: list of documents to encode
        :param normalize_embeddings: whether to normalize the embeddings using L2 normalization
            if not provided, uses the default setting
        :param show_progress_bar: this parameter is ignored, present for compatibility
        :return: list of embeddings
        """
        embeddings = self.embedding_service.embed(
            text_chunks=docs,
            normalize=normalize_embeddings
        )

        return embeddings


class GenerateTopicNameResponse(BaseModel):
    """
    Schema for the response from the topic name generation LLM.
    """
    explanation: str = Field(..., description="Explanation of the topic name.")
    name: str = Field(..., description="Name given to the topic.")


class GenerateTopicDescriptionResponse(BaseModel):
    """
    Schema for the response from the topic description generation LLM.
    """
    description: str = Field(..., description="Description of the topic.")


class FastTopicDiscovery(BaseTopicer, ConfigurableMixin, CreatableMixin):
    """
    Performs topic discovery using FastTopic library (https://github.com/bobxwu/FASTopic).
    """
    n_topics: int = ConfigurableValue(desc="Number of topics to discover.", user_default=10)

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
    topic_doc_search_time: int = ConfigurableValue(
        desc="Number of top documents per topic to search for time phrases when generating topic descriptions.",
        user_default=100,
    )
    topic_time_doc_rep_size: int = ConfigurableValue(
        desc="Number of documents with time phrases to represent each topic when generating topic descriptions. If more documents with time phrases are found, this number will be randomly sampled from them.",
        user_default=3,
    )
    generate_topic_name_model: str = ConfigurableValue(
        desc="Model to use for generating topic names.",
        user_default="gpt-5-mini",
    )
    generate_description_model: str = ConfigurableValue(
        desc="Model to use for generating topic descriptions.",
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

    generate_description_system_prompt: Template = ConfigurableValue(
        desc="System prompt for generating topic descriptions based on time phrases.",
        user_default=LiteralScalarString("""Vytvoř popis tématu na základě názvu tématu, dokumentů reprezentujících téma a dokumentů obsahujících časové výrazy.
Ve svém popisu se zaměř na časové události a jejich význam v kontextu tématu.

Příklad:

Název tématu: Středověká Evropa

Dokumenty reprezentující téma:
Praha byla významným centrem během středověku.
Proběhlo mnoho důležitých událostí v Evropě během středověku.

Dokumenty s časovými výrazy:
Dokument 1: Karel IV. se narodil ve 14. století.
Dokument 2: Bitva u Hastings v roce 1066 byla klíčovou událostí.

Popis tématu:
Téma "Středověká Evropa" zahrnuje období od 5. do 15. století, které bylo svědkem významných historických událostí, jako je narození Karla IV. ve 14. století a Bitva u Hastings v roce 1066.

Vrať pouze popis tématu bez dalších vysvětlení ve formátu JSON:
{
    "description": "Popis tématu"
}

Důležité: Výstupem musí být pouze tento JSON, bez dalšího textu.
Důležité: Cílit na časové události a jejich význam v kontextu tématu.
"""),
        transform=TemplateTransformer()
    )

    generate_description_prompt: Template = ConfigurableValue(
        desc="Jinja2 template prompt for generating topic descriptions based on time phrases.",
        user_default=LiteralScalarString("""Název tématu: {{ topic_name }}
Dokumenty reprezentující téma:
{% for doc in topic_docs %}
{{doc.text}}
{% endfor %}
Dokumenty s časovými výrazy:
{% for doc in time_docs %}
{{doc.text}}
{% endfor %}

Popis tématu:"""),
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

    time_text_detector: TimeTextDetector = ConfigurableSubclassFactory(
        TimeTextDetector,
        desc="Detector for finding time expressions in text.",
        user_default=CzechTimeTextDetector,
        voluntary=True
    )

    def set_embedding_service(self, embedding_service: 'BaseEmbeddingService') -> None:
        self.embedding_service = embedding_service
        self.embedding_service_wrapper = EmbeddingServiceWrapper(embedding_service)

    def check_init(self) -> None:
        """Check if all required services are set. Raise MissingServiceError if not."""
        if self.llm_service is None:
            raise MissingServiceError("LLM service is not set.")

        if self.embedding_service is None:
            raise MissingServiceError("Embedding service is not set.")

        # db_connection is optional for topic discovery

    async def discover_topics_sparse(self, texts: Sequence[TextChunk], n: int | None = None) -> DiscoveredTopicsSparse:
        texts = self.truncate_texts(texts, self.max_char_length)
        top_words, doc_topic_dist = await self._get_topics(texts=texts, n=n)
        return await self._process_topics(
            top_words=top_words,
            doc_topic_dist=doc_topic_dist,
            texts=texts,
            sparse=True
        )

    async def discover_topics_dense(self, texts: Sequence[TextChunk], n: int | None = None) -> DiscoveredTopics:
        texts = self.truncate_texts(texts, self.max_char_length)
        top_words, doc_topic_dist = await self._get_topics(texts=texts, n=n)
        return await self._process_topics(
            top_words=top_words,
            doc_topic_dist=doc_topic_dist,
            texts=texts,
            sparse=False
        )

    async def discover_topics_in_db_sparse(self, db_request: DBRequest, n: int | None = None) -> DiscoveredTopicsSparse:
        if self.db_connection is None:
            raise MissingServiceError("DB connection has to be set for DB topic discovery.")

        texts = self.db_connection.get_text_chunks(db_request)
        return await self.discover_topics_sparse(texts=texts, n=n)

    async def discover_topics_in_db_dense(self, db_request: DBRequest, n: int | None = None) -> DiscoveredTopics:
        if self.db_connection is None:
            raise MissingServiceError("DB connection has to be set for DB topic discovery.")

        texts = self.db_connection.get_text_chunks(db_request)
        return await self.discover_topics_dense(texts=texts, n=n)

    async def propose_tags(self, text_chunk: TextChunk, tags: list[Tag]) -> TextChunkWithTagSpanProposals:
        raise NotImplementedError()

    async def propose_tags_in_db(self, tag: Tag, db_request: DBRequest) -> list[TextChunkWithTagSpanProposals]:
        raise NotImplementedError()

    async def _get_topics(self, texts: Sequence[TextChunk], n: int | None = None) -> tuple[list[str], NDArray]:
        """
        Discovers topics using the FASTopic model.

        :param texts: Sequence of TextChunk objects.
        :param n: Optional number of topics to discover. If None, uses the default from
        :return: Tuple containing:
            - List of top words for each topic.
            - Document-topic distribution matrix.
        """

        model = await asyncio.to_thread(self._create_fastopic_model, n=n)

        top_words, doc_topic_dist = await asyncio.to_thread(
            model.fit_transform,
            docs=[text.text for text in texts]
        )
        return top_words, doc_topic_dist

    @staticmethod
    def truncate_texts(texts: Sequence[TextChunk], max_char_length: int) -> Sequence[TextChunk]:
        """
        Truncates texts to the specified maximum character length.

        :param texts: Sequence of TextChunk objects.
        :param max_char_length: Maximum character length for each text.
        :return: Sequence of truncated TextChunk objects. Makes a deep copy of the input texts.
        """
        texts = copy.deepcopy(texts)

        for text_chunk in texts:
            if len(text_chunk.text) > max_char_length:
                text_chunk.text = text_chunk.text[:max_char_length]

        return texts

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
            doc_embed_model=self.embedding_service_wrapper,
            verbose=self.verbose,
            normalize_embeddings=self.embedding_service.normalize_embeddings if hasattr(self.embedding_service, 'normalize_embeddings') else False,
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

        text_chunks = []
        for words, docs in zip(top_words, top_docs_per_topic):
            prompt = self.generate_topic_name_prompt.render({
                "topic_words": words,
                "topic_docs": docs
            })
            text_chunks.append(prompt)


        api_output = await self.llm_service.process_text_chunks_structured(
            text_chunks=text_chunks,
            instruction=self.generate_topic_name_system_prompt.render({}),
            output_type=GenerateTopicNameResponse,
            model=self.generate_topic_name_model
        )

        res = [(output.name, output.explanation) for output in api_output]
        return res

    async def _generate_time_based_topic_descriptions(self, topic_names: list[str],
                                                      top_docs_per_topic: list[list[TextChunk]],
                                                      top_time_docs_per_topic: list[list[TextChunk]]) -> list[str]:
        """
        Generates topic descriptions based on time phrases found in documents.

        :param topic_names: List of topic names.
        :param top_docs_per_topic: List of lists containing top documents for each topic.
        :param top_time_docs_per_topic: List of lists containing top documents with time phrases for each topic.
        :return: List of topic descriptions.
        """

        text_chunks = []
        for name, topic_docs, time_docs in zip(topic_names, top_docs_per_topic, top_time_docs_per_topic):
            prompt = self.generate_description_prompt.render({
                "topic_name": name,
                "topic_docs": topic_docs,
                "time_docs": time_docs
            })
            text_chunks.append(prompt)

        api_outputs = await self.llm_service.process_text_chunks_structured(
            text_chunks=text_chunks,
            instruction=self.generate_description_system_prompt.render({}),
            output_type=GenerateTopicDescriptionResponse,
            model=self.generate_description_model
        )
        topic_descriptions = [output.description for output in api_outputs]
        return topic_descriptions

    async def _get_time_docs_per_topic(self, topic_doc_dist: NDArray, texts: Sequence[TextChunk]) -> list[list[TextChunk]]:
        """
        Retrieves the top documents containing time phrases for each topic.

        :param topic_doc_dist: document distributions of topics (topic-doc distributions), a numpy array with shape A × B (number of topics A and number of documents B ).
        :param texts: Original text chunks.
        :return: List of randomly selected documents with time phrases for each topic.
        """

        top_docs_per_topic = await asyncio.to_thread(
            self.get_top_k_docs_per_topic, topic_doc_dist=topic_doc_dist, texts=texts, k=self.topic_doc_search_time
        )

        time_docs_per_topic = []
        for docs in top_docs_per_topic:
            time_docs = []
            for doc in docs:
                detected_times = self.time_text_detector(doc.text)
                if detected_times:
                    time_docs.append(doc)

            if len(time_docs) <= self.topic_time_doc_rep_size:
                time_docs_per_topic.append(time_docs)
            else:
                time_docs_per_topic.append(random.sample(time_docs, self.topic_time_doc_rep_size))

        return time_docs_per_topic

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

        top_time_docs_per_topic = await self._get_time_docs_per_topic(topic_doc_dist, texts)
        topic_descriptions = await self._generate_time_based_topic_descriptions(
            [name for name, _ in topic_names_with_explanation],
            top_docs_per_topic,
            top_time_docs_per_topic
        )

        topics = [
            Topic(name=name, name_explanation=explanation, description=topic_descriptions[i]) for i, (name, explanation) in enumerate(topic_names_with_explanation)
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
