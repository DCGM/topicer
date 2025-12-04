# topicer

Projekt pro automatické navrhování tagů (štítků) k textům pomocí LLM (Large Language Model). Systém extrahuje relevantní úseky textu a mapuje je na předem definované tagy s přesným určením pozice (span) v textu.

## Přehled projektu

### Hlavní komponenty

- **`topicer/tagging/`** - Jádro projektu
  - `tag_proposals.py` - Hlavní třída `TagProposal` pro návrh tagů pomocí OpenAI API
  - `config.py` - Konfigurace z `config.yaml`
  - `utils.py` - Pomocné funkce (hledání pozic textu, apod.)
  - `schemas.py` - Datové schémata specifická pro tagging

- **`topicer/schemas.py`** - Veřejná schémata
  - `Tag` - Definice tagu (id, name, description)
  - `TextChunk` - Textový úsek
  - `TagSpanProposal` - Návrh tagu s přesnou pozicí v textu
  - `TextChunkWithTagSpanProposals` - Výsledek zpracování

- **`topicer/database/`** - Databázové operace
  - `db_handler.py` - Práce s databází (Weaviate)

- **`tests/`** - Testovací data a skripty
  - `propose_tags/` a `propose_tags2/` - Testovací sady s vstupy a očekávanými výstupy
  - `test_data.py` - Vzorová testovací data

- **`examples/`** - Příklady použití
  - `example.py` - Příklad spuštění
  - `configs/` - Ukázkové konfigurační soubory

## Setup

### Windows

**PowerShell:**
```powershell
# Vytvoření virtual environment
python -m venv venv

# Aktivace
.\venv\Scripts\Activate.ps1

# Instalace závislostí
pip install -r requirements.txt

# Nastavení OpenAI API klíče
echo OPENAI_API_KEY=your-key-here > .env
```

**Command Prompt (cmd.exe):**
```cmd
# Vytvoření virtual environment
python -m venv venv

# Aktivace
venv\Scripts\activate.bat

# Instalace závislostí
pip install -r requirements.txt

# Nastavení OpenAI API klíče
echo OPENAI_API_KEY=your-key-here > .env
```

**Git Bash:**
```bash
# Vytvoření virtual environment
python -m venv venv

# Aktivace
source venv/Scripts/activate

# Instalace závislostí
pip install -r requirements.txt

# Nastavení OpenAI API klíče
echo "OPENAI_API_KEY=your-key-here" > .env
```

### Linux / macOS

```bash
# Vytvoření virtual environment
python3 -m venv venv

# Aktivace
source venv/bin/activate

# Instalace závislostí
pip install -r requirements.txt

# Nastavení OpenAI API klíče
# Vytvoř .env soubor v kořeni projektu:
echo "OPENAI_API_KEY=your-key-here" > .env
```

## Spuštění

```powershell
# Spuštění demo skriptu
python run.py
```

Demo skript:
1. Načte testovací text a tagy z `tests/test_data.py`
2. Zavolá OpenAI API pro návrh tagů
3. Vrátí seznam tagů s přesnými pozicemi v textu (span_start, span_end)
4. Vytiskne výsledky v čitelné formě

## Jak projekt funguje

1. **Vstup**: Textový úsek + seznam definovaných tagů
2. **Zpracování**: LLM vrátí relevantní tagy s přesným textem (quote)
3. **Post-processing**: Python skript najde přesnou pozici v originálním textu
4. **Výstup**: `TextChunkWithTagSpanProposals` s přesným mapováním

### Klíčové soubory pro běh

- `run.py` - Startovací skript
- `config.yaml` - Konfigurace (OpenAI model, granularita, apod.)
- `.env` - Skrytý soubor s OPENAI_API_KEY

## Konfigurace

`config.yaml` obsahuje:
- `model` - OpenAI model (např. `gpt-4o`)
- `reasoning` - Úsilí LLM (`low`, `medium`, `high`)
- `span_granularity` - Velikost hledaných úseků (`word`, `phrase`, `sentence`)

## Formátování kódu

Projekt používá `autopep8` pro formátování Python kódu.

## Závislosti

- `openai` - OpenAI API klient
- `pydantic` - Datová schémata a validace
- `python-dotenv` - Načítání .env proměnných
- `PyYAML` - Parsování config.yaml
- `weaviate-client` - Klient pro vektorovou databázi
