from abc import ABC, abstractmethod
from typing import Sequence
from topicer.schemas import TextChunk, DBRequest, DiscoveredTopics, DiscoveredTopicsSparse


class TopicDiscovery(ABC):
    """
    Base class for topic discovery methods.
    """

    @abstractmethod
    async def discover_topics(self, texts: Sequence[TextChunk], n: int | None = None, sparse: bool = True) -> DiscoveredTopics | DiscoveredTopicsSparse:
        """
        Discover topics for a collection of texts.

        :param texts: Text chunks to propose tags for.
        :param n: Optional number of topics to propose, if None uses the default value.
        :param sparse: Whether to return sparse representation of discovered topics.
        :return: DiscoveredTopics
        """
        ...

    @abstractmethod
    async def discover_topics_in_db(self, db_request: DBRequest, n: int | None = None, sparse: bool = True) -> DiscoveredTopics | DiscoveredTopicsSparse:
        """
        Discover topics based on a database request.

        :param db_request: Database request to fetch texts for topic discovery.
        :param n: Optional number of topics to propose, if None uses the default value.
        :param sparse: Whether to return sparse representation of discovered topics.
        :return: DiscoveredTopics
        """
        ...

