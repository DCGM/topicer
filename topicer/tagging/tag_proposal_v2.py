# varianta - Marek Sucharda

import json
from uuid import UUID, uuid4
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
from topicer.schemas import TextChunk, Tag, TagSpanProposal, TextChunkWithTagSpanProposals
from topicer.tagging.schemas import AppConfig
from typing import Literal
import logging
from topicer.database.db_schemas import DBRequest

class TagProposalV2:
    def __init__(self, config: AppConfig, openai_client: AsyncOpenAI):
        self.config = config
        self.openai_client = openai_client
        
    # funkce využívá běžný client.chat.completions.
    async def propose_tags(self, text_chunk: TextChunk, tags: list[Tag]) -> TextChunkWithTagSpanProposals:
        # 1. tagy do JSONu
        tags_info = json.dumps([tag.model_dump(mode="json")
                               for tag in tags], ensure_ascii=False)

        # 2. systémový prompt
        system_prompt = """
        Jsi důkladný extraktor entit. Tvým úkolem je najít VŠECHNY výskyty zadaných tagů v textu.

        KRITICKÁ PRAVIDLA:
        1. "quote" musí být PŘESNÝ podřetězec ze vstupního textu.
        2. NESLUČUJ VÝSKYTY! Pokud se slovo (např. "Python") v textu vyskytuje 2x, musíš vrátit 2 objekty v poli "matches".
        3. Procházej text slovo po slově a vypiš každý nalezený tag jako samostatnou položku.
        4. Výstup musí být validní JSON objekt: { "matches": [ { "tag_id": "UUID", "quote": "text", "reason": "důvod", "confidence": 0.0-1.0 } ] }
        """

        # 3. uživatelský prompt
        user_prompt = f"""
        VSTUPNÍ TEXT:
        "{text_chunk.text}"

        HLEDANÉ TAGY:
        {tags_info}
        """

        # 4. volání OpenAI
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.config.openai.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],

                # vynucení JSON výstupu
                response_format={"type": "json_object"},
                # pro konzistenci odpovědí
                temperature=0
            )

            # zpracování odpovědi
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Chybná odpověď od OpenAI: content is None")

            # parsování odpovědi do JSONu
            data = json.loads(content)

            # extrakce matches z JSONu - takto to máme definované v promptu
            matches = data.get("matches", [])

            # příprava návrhů
            proposals = []

            # nalezení indexu pro výskyty
            '''
            mohlo by se stát že v textu "Python je super jazyk. Python se mi líbí."
            se hledá tag programovacích jazyků, OpenAI vrátí dvakrát "Python" jako quote.
            Při hledání pomocí .find() bychom ale pořád našli první výskyt.
            Takže se vytvoří slovník search_start_indices.
            Ten sleduje, kde jsme skončili u každého quote a díky tomu se
            správně označí start_index a end_index pro každý výskyt.
            '''

            # cache pro hlídání pozic opakovaných slov
            search_start_indices = {}

            for match in matches:
                tag_id_str = match.get("tag_id")
                # Převod string na UUID a nalezení tagu
                tag_uuid = UUID(tag_id_str)
                matching_tag = next((tag for tag in tags if tag.id == tag_uuid), None) # next pouzivame pro rychlejsi nalezeni, protoze nemusime projit cely seznam
                
                # Kontrola, zda byl tag nalezen
                if matching_tag is None:
                    logging.warning(f"Tag s UUID {tag_uuid} nebyl nalezen v poskytnutém listu tagů")
                    continue  # přeskočit tento match

                quote = match.get("quote")

                if matching_tag and quote:
                    # odkud máme hledat (abychom nenašli pořád to stejné první slovo)
                    search_from = search_start_indices.get(quote, 0)
                    start_index = text_chunk.text.find(quote, search_from)

                    if start_index != -1:
                        end_index = start_index + len(quote)

                        # uložení nové pozice pro příští hledání stejného slova
                        search_start_indices[quote] = end_index

                        proposals.append(TagSpanProposal(
                            tag=matching_tag,
                            span_start=start_index,
                            span_end=end_index,
                            # confidence=TODO
                            reason=match.get("reason", "Nalezeno modelem")
                        ))

            # vrácení výsledku ve chtěném formátu
            return TextChunkWithTagSpanProposals(
                id=text_chunk.id,
                text=text_chunk.text,
                tag_span_proposals=proposals,
            )

        except Exception as e:
            print(f"Chyba v propose_tags2: {e}")
            return TextChunkWithTagSpanProposals(
                id=text_chunk.id,
                text=text_chunk.text,
                tag_span_proposals=[],
            )
            
    async def propose_tags_in_db(self, tag: Tag,  db_request: DBRequest) -> list[TextChunkWithTagSpanProposals]:
        pass
    
if __name__ == "__main__":
    from tests.test_data import text_chunk, tag1, tag2, tag3

    # načtení API klíče
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        print("Chyba: OPENAI_API_KEY není nastaven v .env souboru")
        exit(1)

    if __debug__:
        print("API Key loaded" if API_KEY is not None else "API Key not found")

    openai_client = AsyncOpenAI(api_key=API_KEY)

    tag_proposal = TagProposalV2("config.yaml", openai_client)

    # run tag proposal
    '''
    # propose_tags is async; run it with asyncio
    tag_array = [tag1, tag2, tag3]
    result = asyncio.run(tag_proposal.propose_tags(text_chunk, tag_array))
    print(result.model_dump_json(indent=4))

    converted_tags = [tag.model_dump(mode="json") for tag in tag_array]
    print(json.dumps(converted_tags, indent=4))
    '''

    async def test_run_propose_tags(variant: Literal[1, 2, 3]):
        chunk = TextChunk(
            id=uuid4(),
            text="Java je fajn, ale Python je lepší. Python se používá pro AI a backend."
        )
        tag_list = [
            Tag(
                id=uuid4(),
                name="Programovací jazyk",
                description="Názvy programovacích jazyků (Java, C++, Python...)"
            ),
            Tag(
                id=uuid4(),
                name="Technologie",
                description="Obecné IT pojmy jako AI, backend, frontend."
            )
        ]

        if __debug__:
            print("\nVstupní text:", chunk.text)
            print("Hledané tagy:", json.dumps([tag.model_dump(
                mode="json") for tag in tag_list], indent=4, ensure_ascii=False), "\n")

        # zavolání propose tags
        if variant == 1:
            # TODO nefunguje
            result = await tag_proposal.propose_tags(chunk, tag_list)
        elif variant == 2:
            result = await tag_proposal.propose_tags2(chunk, tag_list)

        # print výsledků
        if __debug__:
            print("Výsledek:")
            print(result.model_dump_json(indent=4, ensure_ascii=False))

        # zavření klienta OpenAI
        await openai_client.close()

    ### odkomentuj pro spuštění testu ###
    # asyncio.run(test_run_propose_tags(2))