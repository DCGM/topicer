from dotenv import load_dotenv
import os
import logging
from tests.test_data import tag1, tag2, tag3, text_chunk
from topicer.schemas import TextChunkWithTagSpanProposals
from topicer.base import factory
from topicer.embedding.default_service import DefaultEmbeddingService
from topicer.database.weaviate_service import WeaviateService
from topicer.schemas import Tag
from uuid import uuid4


def print_tag_proposals_with_spans(proposals: TextChunkWithTagSpanProposals):
    """Vytiskne tag proposals s vykrojeným textem podle span_start a span_end"""
    print(f"\n{'='*80}")
    print(f"Text ID: {proposals.id}")
    print(f"Text: {proposals.text[:100]}...")
    print(f"{'='*80}\n")

    for i, proposal in enumerate(proposals.tag_span_proposals, 1):
        # Vykrojení textu podle indexů
        span_text = proposals.text[proposal.span_start:proposal.span_end]

        print(f"Návrh {i}:")
        print(f"  Tag: {proposal.tag.name} (ID: {proposal.tag.id})")
        print(f"  Span: [{proposal.span_start}:{proposal.span_end}]")
        print(f"  Text: \"{span_text}\"")
        print(f"  Confidence: {proposal.confidence}")
        print(f"  Reason: {proposal.reason}")
        print()


async def main():
    # načtení API klíče
    load_dotenv()

    # # Factory načte API klíč automaticky z prostředí
    topicer = factory("config.yaml")
    
    # tag: Tag = Tag(
    #     id=uuid4(),
    #     name="Jaká byla návštěva v Belgii v Antverpách v atletických mezinarodních závodech?",
    # )
    tag: Tag = Tag(
        id=uuid4(),
        name="Jak se jmenovalo trojsvazkové dílo, které sjednocovalo soustau měr?",
    )
    
    # proposals: TextChunkWithTagSpanProposals = await topicer.propose_tags(text_chunk, [tag1, tag2, tag3])
    proposals = await topicer.propose_tags_in_db(tag, None)
    
    print(f"\nCelkem nalezeno {len(proposals)} textů s návrhy tagů v DB.\n")

    for proposals in proposals:
        print_tag_proposals_with_spans(proposals)
        

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
