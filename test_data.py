from uuid import uuid4
from tag_proposals import TextChunk, Tag

text_chunk: TextChunk = TextChunk(
    id=uuid4(),
    text="""V moderním světě technologií je informace jedním z nejcennějších zdrojů. Každý den vzniká obrovské množství dat — od běžného používání sociálních sítí, přes transakce finančních institucí, až po senzory monitorující chytré domácnosti či celé městské infrastruktury. Všechna tato data ale sama o sobě nemají velkou hodnotu, dokud je někdo nebo něco nevyhodnotí, nezpracuje a nepřevede na znalosti využitelné v praxi.

Jedním z nejdůležitějších nástrojů, který lidstvu umožňuje tato data efektivně analyzovat, je strojové učení. To umožňuje algoritmům najít vzory, předvídat budoucí události, nebo automatizovat činnosti, které by jinak trvaly člověku neúnosně dlouho. Díky tomu můžeme například detekovat podvodné platby, doporučovat relevantní obsah uživatelům nebo analyzovat zdravotní data s cílem odhalit onemocnění dříve, než se projeví.

Přestože technologie pokročily velmi rychle, stále existuje mnoho výzev. Jednou z nich je etika — kdo přesně vlastní data, kdo je smí zpracovávat a jak zajistit, aby umělá inteligence rozhodovala spravedlivě? Dalším problémem je samotná kvalita dat. Algoritmy mohou být jen tak dobré, jak dobré jsou vstupy, ze kterých se učí. Pokud jsou data zkreslená, neúplná nebo nereprezentativní, výsledky budou nepřesné nebo rizikové.

Budoucnost zpracování dat tedy nespočívá pouze v zlepšování algoritmů, ale také v přístupech, které zdůrazňují odpovědné nakládání s informacemi, transparentnost a respekt k soukromí. Pokud se tyto principy podaří prosadit, technologie může výrazně přispět k lepšímu fungování společnosti — ať už v oblasti zdravotnictví, vzdělávání, průmyslu nebo chytrých měst.""")

tag1: Tag = Tag(
    id=uuid4(),
    name="Strojové učení",
)
tag2: Tag = Tag(
    id=uuid4(),
    name="Etika v AI",
)
tag3: Tag = Tag(
    id=uuid4(),
    name="Big Data",
)
