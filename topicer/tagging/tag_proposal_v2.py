# varianta - Marek Sucharda

import json
import logging
from uuid import UUID
from typing import Optional

from classconfig import ConfigurableMixin, ConfigurableValue
from pydantic import BaseModel, Field
from transformers import pipeline

from topicer.base import BaseTopicer, MissingServiceError
from topicer.schemas import TextChunk, Tag, TagSpanProposal, TextChunkWithTagSpanProposals, DBRequest


class TagProposalMatch(BaseModel):
    tag_id: UUID
    quote: str
    reason: str | None = None
    confidence: float


class TagProposalResponse(BaseModel):
    matches: list[TagProposalMatch] = Field(default_factory=list)


class TagProposalV2(BaseTopicer, ConfigurableMixin):
    model: str = ConfigurableValue(
        desc="Zero-shot classification model from HuggingFace",
        user_default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    )
    device: int = ConfigurableValue(
        desc="Device (0 for GPU, -1 for CPU)", user_default=0)
    tag_proposal_threshold: float = ConfigurableValue(
        desc="Confidence threshold for accepting a tag proposal", user_default=0.6)
    find_probable_tags_threshold: float = ConfigurableValue(
        desc="Confidence threshold for zero-shot classification", user_default=0.6)

    def check_init(self) -> None:
        if self.llm_service is None:
            raise MissingServiceError(
                "LLM service is not set for TagProposalV2.")
        if self.db_connection is None:
            raise MissingServiceError(
                "DB connection is not set for TagProposalV2.")

        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.model,
            device=self.device
        )

    def find_most_probable_tag(self, text: str, tags: list[Tag]) -> Optional[dict]:
        if not text or not tags:
            return None

        tag_names = [tag.name for tag in tags]
        result = self.classifier(text, tag_names)

        best_label = result['labels'][0]  # type: ignore
        best_score = result['scores'][0]  # type: ignore

        best_tag_obj = next((t for t in tags if t.name == best_label), None)

        if best_tag_obj and best_score >= self.find_probable_tags_threshold:
            return {
                "tag": best_tag_obj,
                "confidence": best_score
            }
        return None

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

        # 4. volání OpenAsyncAPI

        try:
            responses = await self.llm_service.process_text_chunks_structured(
                text_chunks=[user_prompt],
                instruction=system_prompt,
                output_type=TagProposalResponse
            )

            matches = responses[0].matches if responses else []

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
                tag_uuid = match.tag_id

                # next pouzivame pro rychlejsi nalezeni, protoze nemusime projit cely seznam
                matching_tag = next(
                    (tag for tag in tags if tag.id == tag_uuid), None)

                # Kontrola, zda byl tag nalezen
                if matching_tag is None:
                    logging.warning(
                        f"Tag s UUID {tag_uuid} nebyl nalezen v poskytnutém listu tagů")
                    continue  # přeskočit tento match

                quote = match.quote

                if matching_tag and quote:
                    # odkud máme hledat (abychom nenašli pořád to stejné první slovo)
                    search_from = search_start_indices.get(quote, 0)
                    start_index = text_chunk.text.find(quote, search_from)

                    if start_index != -1:
                        end_index = start_index + len(quote)
                        # uložení nové pozice pro příští hledání stejného slova
                        search_start_indices[quote] = end_index

                        # kontrola thresholdu
                        if match.confidence >= self.tag_proposal_threshold:
                            proposals.append(TagSpanProposal(
                                tag=matching_tag,
                                span_start=start_index,
                                span_end=end_index,
                                confidence=match.confidence,
                                reason=match.reason or "Nalezeno modelem"
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
