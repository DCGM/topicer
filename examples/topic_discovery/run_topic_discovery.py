#!/bin/env python3
import argparse
import asyncio
import uuid
from pathlib import Path

from classconfig import Config

from topicer.base import TopicerFactory, factory
from topicer.schemas import TextChunk
from topicer.topic_discovery import FastTopicDiscovery

SCRIPT_DIR = Path(__file__).parent

RAW_TEXTS = [
    # Matematika
    "Zlatý řez: Tento matematický poměr, často označovaný řeckým písmenem fí, se v přírodě vyskytuje často, od uspořádání listů na stonku až po spirálu ulity loděnky, a vytváří esteticky příjemné proporce.",
    "Povaha nuly: Pojetí nuly jako samostatného čísla – nikoli jen jako zástupného symbolu – bylo revolučním vývojem v matematice, který se poprvé objevil ve staroindických textech kolem 5. století.",
    "Prvočísla v přírodě: Některé druhy cikád se líhnou pouze jednou za 13 nebo 17 let; evoluční biologové se domnívají, že tyto prvočíselné životní cykly jim pomáhají vyhnout se predátorům, kteří mají kratší a předvídatelnější populační nárůsty.",
    "Nekonečná hodnota čísla pí: Pí (π) je iracionální číslo, což znamená, že jeho desetinný rozvoj nikdy nekončí a neustálí se v trvale se opakujícím vzorci, přičemž bylo vypočítáno na biliony cifer.",
    "Fraktální geometrie: Na rozdíl od běžné geometrie jsou fraktály složité obrazce, které jsou soběpodobné v různých měřítkách, což znamená, že pokud přiblížíte část útvaru, vypadá podobně jako celek.",
    "Narozeninový paradox: V místnosti s pouhými 23 lidmi je ve skutečnosti více než 50% šance, že dva lidé budou mít narozeniny ve stejný den, což je výsledek, který často odporuje lidské intuici ohledně pravděpodobnosti.",
    "Pythagoras a hudba: Pythagoras není známý jen svou větou o trojúhelnících, ale také objevem, že harmonické hudební intervaly jsou tvořeny poměry celých čísel.",

    # Historie
    "Kleopatřina časová osa: Kleopatra žila v době, která je časově blíže přistání na Měsíci než stavbě Velké pyramidy v Gíze, což zdůrazňuje nesmírnou délku trvání staroegyptské civilizace.",
    "Nejkratší válka: Anglo-zanzibarská válka z roku 1896 je zaznamenána jako nejkratší válka v historii, trvala 38 až 45 minut, než bylo vyhlášeno příměří.",
    "Malta Velké čínské zdi: Stavební dělníci během dynastie Ming používali k pojení cihel Velké čínské zdi maltu vyrobenou z polévky z lepkavé rýže smíchané s hašeným vápnem, což zajišťovalo neuvěřitelnou trvanlivost.",
    "Napoleonova výška: Představa, že byl Napoleon Bonaparte malý, je z velké části mýtus způsobený rozdílem mezi francouzským a britským měrným systémem; na svou dobu byl ve skutečnosti průměrně vysoký.",
    "Knihtisk: Vynález knihtisku s pohyblivými literami Johannese Gutenberga v 15. století drasticky snížil cenu knih, což podpořilo šíření gramotnosti a protestantskou reformaci.",
    "Vikingské helmy: Navzdory populárním vyobrazením ve filmech a seriálech neexistují žádné historické důkazy o tom, že by vikingští bojovníci nosili v boji rohaté helmy; tato představa pochází z velké části z operních kostýmů 19. století.",
    "Alexandrijská knihovna: Alexandrijská knihovna, kdysi největší archiv vědění ve starověkém světě, nebyla zničena jediným požárem, ale pravděpodobně upadala po staletí kvůli nedostatku financí a politické nestabilitě.",

    # Jídlo
    "Trvanlivost medu: Med je jednou z mála potravin, která obsahuje všechny látky nezbytné k udržení života, včetně enzymů, vitamínů a vody; navíc správně uzavřený med se v podstatě nikdy nezkazí.",
    "Cena šafránu: Šafrán je podle váhy nejdražším kořením na světě, protože se získává z jemných blizen květu šafránu setého, které se musí sklízet ručně.",
    "Náhražky wasabi: Většina „wasabi“ podávaného mimo špičkové restaurace je ve skutečnosti směs křenu, hořčice a zeleného potravinářského barviva, protože pěstování pravého kořene wasabi je neuvěřitelně obtížné.",
    "Původ croissantu: Ačkoli je croissant známý jako francouzská klasika, ve skutečnosti vznikl v Rakousku jako „kipferl“, pečivo ve tvaru půlměsíce vytvořené na oslavu porážky Osmanské říše ve Vídni.",
    "Čokoláda jako platidlo: Staří Mayové a Aztékové si kakaových bobů cenili natolik, že je používali jako měnu k nákupu jídla a dalšího zboží.",
    "Pizza Margherita: Legenda praví, že pizza Margherita byla vytvořena v roce 1889 na počest italské královny Markéty Savojské a obsahovala barvy italské vlajky: červená rajčata, bílou mozzarellu a zelenou bazalku."
]

CHUNK_LIST = [
    TextChunk(
        id=uuid.uuid4(),
        text=t
    ) for t in RAW_TEXTS
]


async def call_run(args):
    """
    Method for running Fast Topic Discovery.

    :param args: User arguments.
    """

    topic_discovery = factory(SCRIPT_DIR / "config.yaml")

    res = await topic_discovery.discover_topics_dense(
        texts=CHUNK_LIST,
        n=3
    )
    print("Discovered Topics:")
    for idx, topic in enumerate(res.topics):
        print(f"Topic {idx + 1}:")
        print(topic.model_dump_json(indent=4))

    print("Topic documents assignment:")
    for i, doc_topic in enumerate(res.topic_documents):
        print(f"Topic {i + 1}:")
        print(f"Documents: {doc_topic}")

    print("\nRunning with sparse=True...\n")
    res_sparse = await topic_discovery.discover_topics_sparse(
        texts=CHUNK_LIST,
        n=3
    )
    print("Discovered Topics (sparse):")
    for idx, topic in enumerate(res_sparse.topics):
        print(f"Topic {idx + 1}:")
        print(topic.model_dump_json(indent=4))

    print("Topic documents assignment (sparse):")
    for i, doc_topic in enumerate(res_sparse.topic_documents):
        print(f"Topic {i + 1}:")
        print(f"Documents: {doc_topic}")


async def call_create_config(args):
    """
    Method for creating configuration for Fast Topic Discovery.

    :param args: User arguments.
    """
    # Placeholder for actual implementation
    c = Config(TopicerFactory)
    c.save(SCRIPT_DIR / "config.yaml")
    c.to_md(SCRIPT_DIR / "config.md")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Fast Topic Discovery on a set of documents.")
    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser('run', help="Run Fast Topic Discovery")
    run_parser.set_defaults(func=call_run)

    create_config_parser = subparsers.add_parser('create_config', help="Create configuration for Fast Topic Discovery")
    create_config_parser.set_defaults(func=call_create_config)

    return parser.parse_args()


async def main():
    args = parse_arguments()
    if hasattr(args, 'func'):
        await args.func(args)
    else:
        print("No command provided. Use -h for help.")


if __name__ == "__main__":
    asyncio.run(main())