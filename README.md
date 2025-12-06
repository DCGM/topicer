# topicer

Automatické navrhování tagů (štítků) k textům pomocí LLM. Model označí relevantní úseky textu a přiřadí je k předem definovaným tagům včetně přesných pozic (span_start, span_end).

## Přehled

- **Jádro (`topicer/tagging/`)**
  - `tag_proposal_v1.py` – varianta s `Responses API`, která vrací quote + context a v Pythonu hledá přesné indexy.
  - `tag_proposal_v2.py` – varianta s `Chat Completions API`; hledá opakované výskyty podle posunutého startu.
  - `config.py` – načtení `config.yaml` do `AppConfig` (openai/weaviate).
  - `utils.py` – pomocné funkce, např. robustní hledání spanů podle quote + context.
  - `schemas.py` – interní schémata pro LLM návrhy a konfiguraci.

- **Veřejná schémata (`topicer/schemas.py`)**
  - `Tag`, `TextChunk`, `TagSpanProposal`, `TextChunkWithTagSpanProposals`.

- **Databáze (`topicer/database/`)**
  - Připravené schéma pro Weaviate (`db_schemas.py`, klient `weaviate_client.py`).

- **Příklady a testy**
  - `run.py` – demo s daty z `tests/test_data.py`.
  - `examples/` – ukázky konfigurace a použití.
  - `tests/propose_tags*` – vstupní/výstupní JSONy pro manuální porovnání.

## Struktura projektu

```
topicer/
├── tagging/                          # Jádro pro návrh tagů
│   ├── tag_proposal_v1.py           # Varianta s Responses API
│   ├── tag_proposal_v2.py           # Varianta s Chat Completions API
│   ├── config.py                    # Načtení konfigurace z YAML
│   ├── tagging_schemas.py           # Interní Pydantic schémata
│   └── utils.py                     # pomocné funkce
├── database/                        # Databázová vrstva (Weaviate)
│   ├── db_schemas.py                # DBRequest a další DB schémata
│   └── weaviate_client.py           # Klient pro Weaviate
├── schemas.py                       # Veřejná schémata (Tag, TextChunk, TagSpanProposal)

config.yaml                           # Konfigurace (openai, weaviate)
run.py                                # Vstupní skript – demo
requirements.txt                      # Python závislosti
pyproject.toml                        # Project metadata
.env                                  # OpenAI API klíč (gitignore)

tests/
├── test_data.py                     # Vzorová testovací data
├── propose_tags/                    # Testovací sada V1
│   ├── script.py
│   ├── inputs/                      # test1_cities.json, test2_*.json, …
│   └── outputs/                     # expected_output JSONy
└── propose_tags2/                   # Testovací sada V2
    ├── script.py
    ├── inputs/
    └── outputs/

examples/
├── example.py                        # Příklad základního použití
└── configs/
    └── example_config.yaml

ssh_tunnel_setup/                     # SSH tunelovací skripty
├── config.ini                        # Konfigurace pro Python verzi
├── config.sh                         # Konfigurace pro Bash verzi
├── start_tunnel.py
└── start_tunnel.sh

docs/
└── documentation.md                 # [Bude doplneno]
```

## Rychlý start

### Instalace

#### Windows – PowerShell

```powershell
# vytvoření virtual environment
python -m venv venv

# aktivace
.\venv\Scripts\Activate.ps1

# instalace závislostí
pip install -r requirements.txt

# nastavení OpenAI API klíče
echo OPENAI_API_KEY=your-key-here > .env
```

#### Windows – Command Prompt (cmd)

```cmd
# vytvoření virtual environment
python -m venv venv

# aktivace
venv\Scripts\activate.bat

# instalace závislostí
pip install -r requirements.txt

# nastavení OpenAI API klíče
echo OPENAI_API_KEY=your-key-here > .env
```

#### Windows – Git Bash

```bash
# vytvoření virtual environment
python -m venv venv

# aktivace
source venv/Scripts/activate

# instalace závislostí
pip install -r requirements.txt

# nastavení OpenAI API klíče
echo "OPENAI_API_KEY=your-key-here" > .env
```

#### Linux / macOS

```bash
# vytvoření virtual environment
python3 -m venv venv

# aktivace
source venv/bin/activate

# instalace závislostí
pip install -r requirements.txt

# nastavení OpenAI API klíče
echo "OPENAI_API_KEY=your-key-here" > .env
```

### Spuštění

#### Windows – PowerShell / cmd

```powershell
python run.py
```

#### Linux / macOS

```bash
python3 run.py
```

Skript načte vzorový text a tagy z `tests/test_data.py`, zavolá variantu TagProposal uvedenou v `run.py` a vytiskne návrhy včetně spanů.

## Volba varianty TagProposal

- **V2** – používá `chat.completions` a pokrývá vícenásobné výskyty stejného slova. Vrací výsledky rychleji.
- **V1** – používá `responses.parse` a přesnější dohledání spanů podle `quote + context_before`.

Do budoucna se počítá s volbou varianty přímo v konfiguraci. Do té doby lze přepnout ručně v `run.py` změnou importu:

```python
from topicer.tagging.tag_proposal_v1 import TagProposalV1 as TagProposal
# from topicer.tagging.tag_proposal_v2 import TagProposalV2 as TagProposal
```

Obě varianty vrací `TextChunkWithTagSpanProposals` a očekávají `AppConfig` z `config.yaml` + instanci `AsyncOpenAI`.

## Konfigurace (`config.yaml`)

- `openai.model` – výchozí `gpt-5.1`.
- `openai.reasoning` – `none|low|medium|high` (předává se do V1; V2 běží s temperature 0).
- `openai.span_granularity` – `word|phrase|collocation|sentence|paragraph` (hint pro V1 při volbě délky citace).
- `weaviate.host|rest_port|grpc_port` – připraveno pro napojení na Weaviate.

## SSH tunel

Konfigurace a start skriptů v `ssh_tunnel_setup/` (`config.ini`, `config.sh`, `start_tunnel.py`, `start_tunnel.sh`). Tunel přesměruje např. porty 9000 → 8080 a 50055 → 50051 dle konfigurace.

## Jak to funguje

1. Vstup: `TextChunk` + seznam `Tag`.
2. LLM vrátí návrhy (quote + metadata).
3. Post-processing v Pythonu dopočítá `span_start`/`span_end`.
4. Výstup: `TextChunkWithTagSpanProposals` připravený pro další zpracování nebo uložení.

## Poznámky k vývoji

- Potřebný `.env` s `OPENAI_API_KEY` v kořeni projektu.
- Kód je formátovaný `autopep8`.
- Závislosti: `openai`, `pydantic`, `python-dotenv`, `PyYAML`, `weaviate-client`.
