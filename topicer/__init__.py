from .base import factory
from .tagging.tag_proposal_v1 import TagProposalV1
from .tagging.tag_proposal_v2 import TagProposalV2
from .tagging.gliner import GlinerTopicer

__all__ = [
    "factory",
]
