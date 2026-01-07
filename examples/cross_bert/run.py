from pathlib import Path
import uuid
import asyncio

from topicer.base import factory
from topicer.schemas import TextChunk, Tag


config_path = Path(__file__).parent / "config.yaml"
topicer = factory(str(config_path))

text_chunk = TextChunk(
    id=uuid.uuid4(),
    text="porace již nyní pracují ku hojné účasti na sjezdu a to pokud lze již na poradách v odborech."
    + " Obchodní a živnostenská komora v Brně konala 13. října plenární schůzi, na níž podána zpráva o"
    + " hospodářských poměrech, o nichž komora zavedla podrobné šetření a předložila výsledky šetření v"
    + " obšírném pamětním spisu všem povolaným místům s prosbou, aby přidělením dodávek, nejrychlejším"
    + " provedením připravovaných projektů komornímu obvodu opatřila příležitost k zaměstnání. Též byla"
    + " c. k. vláda požádána, aby na nynější hospodářskou situaci při vyměřování a vymáhání daní nejmožnější"
    + " šetření zřetel vzala. Komora zvolila soudci-laiky opětně E. Tilla a Rud. Zöllnera a nominovala zástupce"
    + " své do školních výborů škol pokračovacích. Po té podán referát o nepřijatelné trattě, o níž podáme"
)

tags = [
    Tag(
        id=uuid.uuid4(),
        name="Export",
        description="Problematika vývozu, účasti domácích výrobků na zahraničních výstavách a organizace vývozních spolků..",
    ),
    Tag(
        id=uuid.uuid4(),
        name="Dozor",
        description="Inspekce a výkon předpisů živnostenských úřadů, kontrola hospodaření a návrhy na úpravu kontrolních pravomocí.",
    ),
    Tag(
        id=uuid.uuid4(),
        name="Družstva",
        description="Konsumy a spotřební družstva, jejich postavení vůči ostatním obchodníkům a snahy o legislativní rovnost.",
    ),
    Tag(
        id=uuid.uuid4(),
        name="Housenky",
        description="Populace housenek jako škůdců na listech a poupatech; metody boje zahrnují ruční sběr, postřiky (tabákové,"
        + " chlorové, chilský ledek), nasazení pasti a podpůrných rostlin (konopí) a ochranu korun stromů lepovými pásy.",
    )
]

async def get_result():
    result = await topicer.propose_tags(
        text_chunk=text_chunk,
        tags=tags,
    )

    return result


if __name__ == "__main__":
    result = asyncio.run(get_result())
    for span in result.tag_span_proposals:
        print(f"{span.tag.name.capitalize()}:")
        print(f"  Span (start: {span.span_start}, end: {span.span_end}): '{text_chunk.text[span.span_start:span.span_end]}'")
        print()
