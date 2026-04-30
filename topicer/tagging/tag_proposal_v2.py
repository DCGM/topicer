# varianta - Marek Sucharda

import json
import logging
from uuid import UUID
from typing import cast

from classconfig import ConfigurableMixin


from topicer.base import BaseTopicer, MissingServiceError
from topicer.llm.openai import OpenAIService
from topicer.schemas import TextChunk, Tag, TagSpanProposal, TextChunkWithTagSpanProposals, DBRequest


class TagProposalV2(BaseTopicer, ConfigurableMixin):

    @property
    def openai(self) -> OpenAIService:
        return cast(OpenAIService, self.llm_service)

    def check_init(self) -> None:
        if self.llm_service is None:
            raise MissingServiceError(
                "LLM service is not set for TagProposalV2.")
        if self.db_connection is None:
            raise MissingServiceError(
                "DB connection is not set for TagProposalV2.")

    # funkce využívá běžný client.chat.completions.
    async def propose_tags(self, text_chunk: TextChunk, tags: list[Tag]) -> TextChunkWithTagSpanProposals:
        # 1. tagy do JSONu
        tags_info = json.dumps([tag.model_dump(mode="json")
                               for tag in tags], ensure_ascii=False)

        # 2. systémový prompt
        system_prompt = """
        Jsi profesionální lidský anotátor textu. Tvým úkolem je číst text a "zvýrazňovat" v něm pasáže, které se týkají zadaných tagů.

        KRITICKÁ PRAVIDLA:
        1. NEOZNAČUJ POUZE IZOLOVANÁ SLOVA! Hodnota "quote" musí vždy obsahovat CELOU VĚTU, VÍCE VĚT, nebo CELÝ ODSTAVEC, který dává smysl jako kontext pro daný tag.
        2. "quote" musí být ABSOLUTNĚ PŘESNÝ podřetězec ze vstupního textu (včetně původních mezer, velkých písmen a interpunkce). Nesmíš vynechat ani upravit jediné písmeno, jinak text v kódu nenajdeme.
        3. Pokud se jedno téma (tag) probírá ve dvou různých, nesouvisejících odstavcích, vrať 2 objekty v poli "matches".
        4. Výstup musí být validní JSON objekt ve formátu:
        { "matches": [ { "tag_id": "UUID", "quote": "celá věta nebo více vět přesně zkopírovaný z textu", "reason": "důvod", "confidence": 0.0-1.0 } ] }
        """

        # 3. uživatelský prompt
        user_prompt = f"""
        VSTUPNÍ TEXT:
        "{text_chunk.text}"

        HLEDANÉ TAGY (zde máš k dispozici UUID tagů a jejich definice):
        {tags_info}
        """

        # 4. volání OpenAI

        try:
            async with self.openai as openai_service:
                response = await openai_service.client.chat.completions.create(
                    model=self.openai.model,
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
                    raise ValueError(
                        "Chybná odpověď od OpenAI: content is None")

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

                    # next pouzivame pro rychlejsi nalezeni, protoze nemusime projit cely seznam
                    matching_tag = next(
                        (tag for tag in tags if tag.id == tag_uuid), None)

                    # Kontrola, zda byl tag nalezen
                    if matching_tag is None:
                        logging.warning(
                            f"Tag s UUID {tag_uuid} nebyl nalezen v poskytnutém listu tagů")
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
                                confidence=match.get("confidence"),
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
        return []
