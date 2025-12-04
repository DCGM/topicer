from openai import AsyncOpenAI
from topicer.tagging.tag_proposals import TagProposal
from dotenv import load_dotenv
import os
import logging
from topicer.schemas import TextChunk
from uuid import uuid4
from tests.test_data import tag1, tag2, tag3, text_chunk
from topicer.schemas import TextChunkWithTagSpanProposals

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
        print(f"  Tag ID: {proposal.tag_id}")
        print(f"  Span: [{proposal.span_start}:{proposal.span_end}]")
        print(f"  Text: \"{span_text}\"")
        print(f"  Confidence: {proposal.confidence}")
        print(f"  Reason: {proposal.reason}")
        print()

async def main():
    # načtení API klíče
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        logging.error("Chyba: OPENAI_API_KEY není nastaven v .env souboru")
        exit(1)
        
    openai_client = AsyncOpenAI(api_key=API_KEY)
    
    tag_proposal = TagProposal("config.yaml", openai_client)

    proposals: TextChunkWithTagSpanProposals = await tag_proposal.propose_tags(text_chunk, [tag1, tag2, tag3])
    
    # print(proposals.model_dump_json(indent=2))
    print_tag_proposals_with_spans(proposals)
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())