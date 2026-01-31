# Topicer

**Topicer** is a Python-based software framework for **topic discovery** and **semantic tag proposal** in large collections of textual documents.  
It provides a unified Python API, pre-trained models, and REST services for easy deployment and integration.

Topicer is developed within the **semANT – Semantic Explorer of Textual Cultural Heritage** project and is designed with modularity, extensibility, and configurability in mind.

---

## Key Features

- **Topic discovery**
  - Unsupervised discovery of topics in text collections
  - Automatic topic naming and description generation using LLMs
  - Dense and sparse topic–document assignment
- **Tag proposal**
  - Zero-shot and few-shot tagging
  - Span-level localization of tags in text
  - Multiple tagging backends (LLM-based, neural models)
- **Unified architecture**
  - Single factory-based initialization from YAML config
  - Shared abstractions for LLMs, embeddings, and databases
- **Deployment options**
  - Python package API
  - REST API server
  - Docker-based deployment
- **Multilingual focus**
  - Optimized for highly inflective languages (e.g. Czech)

---

## Installation

The latest version can be installed directly from GitHub:

```bash
pip install git+https://github.com/DCGM/topicer
````

Install a specific version:

```bash
pip install git+https://github.com/DCGM/topicer@TAG
```

Available releases are listed at:
[https://github.com/DCGM/topicer/tags](https://github.com/DCGM/topicer/tags)

---

## Usage Overview

Topicer is optimized for ease of use through a **single factory function** that constructs a fully configured method instance from a YAML file.

### Basic Example

```python
from topicer import factory

method = factory("method_config.yaml")

# Propose tags in a text chunk
result = await method.propose_tags(text_chunk, tags)

# Discover topics
topics = await method.discover_topics_dense(text_chunks)
```

Each method instance exposes a **unified API** and is fully self-contained.

---

## Models

You can download the trained models here:

| Model | Topicer  | URL  |
|--------------|----------|-------------|
| Base | CrossBertTopicer | [Download](https://nextcloud.fit.vutbr.cz/s/ppWnowwBE9D6f3H/download) |

## Supported Functionality

### Topic Discovery

* Discovers topics in text collections
* Generates:

  * Topic names
  * Topic descriptions
  * Topic–text assignments
* Supports:

  * **Dense representation** (score for every topic–text pair)
  * **Sparse representation** (only strongly associated pairs)

Implemented using:

* FASTopic
  * You need to download MorphoDiTa model for Czech language processing (available at https://lindat.mff.cuni.cz/repository/items/c2ad9429-4111-482b-998f-5e2fc5a5abdd)
* SentenceTransformers (Gemma2-based embeddings)
* Lemmatization via Morphodita
* LLM-based topic naming and description (e.g. `gpt-5-mini`)

---

### Tag Proposal

Topicer supports multiple tag proposal methods:

#### LLM-based Tagging (`LLMTopicer`)

* Uses external LLMs (OpenAI, Ollama, etc.)
* Identifies and localizes tag spans using a robust matching strategy
* Suitable for zero-shot and few-shot tagging

#### GLiNER-based Tagging (`GlinerTopicer`)

* Uses pretrained multilingual GLiNER models
* Token-span based predictions with configurable thresholds
* Supports single-label and multi-label modes

#### Cross-Encoder BERT Tagging (`CrossBertTopicer`)

* Fine-tuned BERT cross-encoder models
* Token-level scoring with span merging
* Optimized for Czech-language tagging
* Pretrained models distributed with the package

---

## API Reference

All topicer methods implement a subset of the following async API:

```python
async def discover_topics_sparse(texts, n=None)
async def discover_topics_dense(texts, n=None)

async def discover_topics_in_db_sparse(db_request, n=None)
async def discover_topics_in_db_dense(db_request, n=None)

async def propose_tags(text_chunk, tags)
async def propose_tags_in_db(tag, db_request)
```

If a method does not implement a requested function, a `NotImplemented` exception is raised.

---

## Configuration

Topicer uses **YAML-based configuration** powered by `classconfig`.

### Configuration Structure

```yaml
topicer:
  cls: ClassNameImplementingATopicerMethod
  config:
    # method-specific parameters

llm_service:
  cls: ClassNameImplementingALLMService
  config:
    # LLM parameters

embedding_service:
  cls: ClassNameImplementingAnEmbeddingService
  config:
    # embedding parameters

db_connection:
  cls: ClassNameImplementingADatabaseService
  config:
    # database parameters
```

The configuration is parsed and validated automatically by the factory.

---

## REST API

Topicer includes a REST API built with **FastAPI**.

### Available Endpoints

* `/v1/configs`
* `/v1/topics/discover/texts/sparse`
* `/v1/topics/discover/texts/dense`
* `/v1/topics/discover/db/sparse`
* `/v1/topics/discover/db/dense`
* `/v1/tags/propose/texts`
* `/v1/tags/propose/db`

Swagger documentation is available at:

```
http://localhost:8000/docs
```

---

## Running the API Server

### Local Execution

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r topicer_api/requirements.txt
export PYTHONPATH=`pwd`:$PYTHONPATH
export TOPICER_API_CONFIGS_DIR=`pwd`/deploy/data/configs
cd topicer_api
python run.py
```

Environment variables:

* `APP_HOST` (default: `127.0.0.1`)
* `APP_PORT` (default: `8000`)
* `TOPICER_API_CONFIGS_DIR`
* `TOPICER_API_CONFIGS_EXTENSION`

---

### Docker Deployment

```bash
cd deploy
./update.sh build
./update.sh up
```

The API will be available at:

```
http://localhost:8080
```

GPU acceleration requires **NVIDIA Container Toolkit**.
For CPU-only deployment, comment out the GPU section in `compose.yaml`.

---

## Repository Structure

```
deploy/              Docker deployment
docs/                Method documentation
examples/            Usage examples
tests/               Unit tests
embedding_service/   Embedding REST API
topicer_api/         REST API server
topicer/
  ├─ database/       Database abstractions
  ├─ embedding/      Embedding services
  ├─ llm/            LLM service integrations
  ├─ tagging/        Tag proposal methods
  ├─ topic_discovery/Topic discovery methods
  ├─ utils/
  ├─ base.py         Public API & factory
  ├─ schemas.py      Pydantic schemas
  └─ __init__.py
```

---

## Extending Topicer

Topicer is designed to be easily extensible:

* Add new topicers in `tagging/` or `topic_discovery/`
* Implement new shared services in:

  * `llm/`
  * `embedding/`
  * `database/`
* All new classes **must be imported in `topicer/__init__.py`**

For API-level changes, please open a GitHub issue first.

---

## Requirements

* Python **3.12** recommended
* Optional GPU for local inference
* OpenAI API key or local Ollama server for LLM-based methods
* Docker & Docker Compose for containerized deployment

---

## License

**BSD 3-Clause License**

---

## Acknowledgements

This software was developed with financial support from the **Ministry of Culture of the Czech Republic** under the **NAKI III** program
(Project ID: **DH23P03OVV060**).

**Authors**:
Martin Dočekal, Martin Kostelník, Marin Kišš, Richard Juřica, Michal Hradiš
Brno University of Technology (VUT Brno)

---

