# Topicer – Package Documentation

This document complements the high-level overview in [README.md](../README.md) and
focuses on the public Python API, configuration surface, and the REST endpoints
exposed by `topicer_api`. It reflects the current state of the
`feature--added-new-features-and-improve-the-performance` branch.

---

## 1. Architecture overview

Topicer is organized around a small set of pluggable components:

| Layer | Module | Role |
|-------|--------|------|
| Topicers | `topicer.tagging.*`, `topicer.topic_discovery.*` | High-level methods (tagging, topic discovery) |
| LLM services | `topicer.llm.*` | Async wrappers around OpenAI / Ollama / generic HTTP LLMs |
| Embedding services | `topicer.embedding.*` | Local or remote sentence-embedding services |
| Database connections | `topicer.database.*` | Storage / retrieval back-ends (currently Weaviate) |
| REST API | `topicer_api` | FastAPI server exposing topicers over HTTP |

A single YAML file describes which classes are instantiated and how they are wired
together. The `topicer.factory()` helper parses the YAML, validates it via
`classconfig`, and returns a fully-initialized topicer instance.

---

## 2. Core schemas (`topicer.schemas`)

### `Tag`

```python
class Tag(BaseModel):
    id: UUID
    name: str
    description: str | None = None
    examples: list[str] | None = None
```

* `description` – free-form natural-language definition of the tag.
  Used both in LLM prompts and as part of the embedding query.
* `examples` – list of plain example strings (phrases that should be tagged).
  **Note:** in earlier versions this field was `list[TextWithSpan]`; the span
  information was dropped because it was not used and complicated client code.

### `DBRequest`

```python
class DBRequest(BaseModel):
    collection_id: UUID | None = None
    document_id:   UUID | None = None
```

`DBRequest` narrows the set of chunks a database-backed method operates on:

* `collection_id` – restrict to chunks belonging to a given user collection.
* `document_id` – restrict to chunks of a single document.
* Both fields may be combined; filters are AND-ed together.
* `None` on a field disables that filter.

### `TagSpanProposal` / `TextChunkWithTagSpanProposals`

Standard outputs of all `propose_tags*` calls. `TagSpanProposal` carries the
matched `Tag`, character span, confidence, and an optional explanation.

---

## 3. Topicer Python API

All topicers implement a subset of the following async surface:

```python
async def discover_topics_sparse(texts, n=None)
async def discover_topics_dense(texts, n=None)

async def discover_topics_in_db_sparse(db_request, n=None)
async def discover_topics_in_db_dense(db_request, n=None)

async def propose_tags(text_chunk, tags)
async def propose_tags_in_db(tag, db_request)
async def propose_tags_in_db_stream(tag, db_request)   # async generator
```

Methods that are not implemented raise `NotImplementedError`.

### `propose_tags_in_db_stream`

A new async generator implemented by `LLMTopicer`. It performs the same retrieval
+ LLM tagging pipeline as `propose_tags_in_db`, but yields each
`TextChunkWithTagSpanProposals` as soon as its LLM call finishes
(`asyncio.as_completed` under the hood). Empty proposals are filtered out.

Typical usage:

```python
async for chunk_with_proposals in method.propose_tags_in_db_stream(tag, db_request):
    handle(chunk_with_proposals)
```

This dramatically reduces time-to-first-result for large collections.

---

## 4. `LLMTopicer` internals

### Prompt design

The prompt instructs the model to:

1. Read the input text.
2. Inspect every available tag (`name`, optional `description`, optional
   `examples`) and use all those fields to understand the tag's meaning.
3. Identify spans that match a tag and report:
   - `quote` – exact substring,
   - `context_before` / `context_after` – 5–10 surrounding words,
   - `tag_id` – the UUID of the matched tag (copied from the Available Tags list),
   - `confidence` – float in `[0, 1]`,
   - `reason` – optional rationale.

### Why `tag_id` instead of a full `Tag`?

Earlier the LLM was asked to echo back the entire `Tag` object. This wasted
tokens and was prone to schema hallucination. The model now returns only the
UUID; the topicer rehydrates the full `Tag` from an in-memory map (`tags_by_id`)
before constructing the `TagSpanProposal`. Unknown ids are logged and skipped.

### Retrieval embedding for `propose_tags_in_db*`

When searching the database for candidate chunks, the topicer no longer embeds
just the tag name. It builds a query string from:

```
tag.name
tag.description           (if set)
tag.examples              (joined line-by-line, if set)
```

This significantly improves retrieval quality, particularly for short or
ambiguous tag names.

---

## 5. `WeaviateService`

`topicer.database.weaviate_service.WeaviateService` is configured via
`ConfigurableValue`s. Defaults can be overridden in YAML.

| Field | Default | Description |
|-------|---------|-------------|
| `host` / `rest_port` / `grpc_port` | – | Connection parameters |
| `chunks_collection` | `Chunks_test` | Collection storing text chunks |
| `documents_collection` | `Documents` | Collection storing parent documents |
| `chunk_user_collection_ref` | `userCollection` | Reference property used to filter by `DBRequest.collection_id` |
| `chunk_document_ref` | `document` | Reference property used to filter by `DBRequest.document_id` |
| `chunk_text_prop` | – | Property holding chunk text |
| `chunks_limit` | – | Default top-k for similarity search |
| `max_vector_distance` | `1.0` | Cosine-distance threshold. `0.0` = identical, `1.0` = unrelated, `2.0` = opposite. Suggested ranges: `0.3–0.5` strict, `0.5–0.7` moderate. Chunks above the threshold are excluded from results. |

`find_similar_text_chunks` builds Weaviate filters from the supplied `DBRequest`:

* `collection_id` set → `Filter.by_ref(chunk_user_collection_ref).by_id().equal(...)`
* `document_id`   set → `Filter.by_ref(chunk_document_ref).by_id().equal(...)`
* Both set → filters AND-ed together.

---

## 6. REST API (`topicer_api`)

FastAPI server, served by `topicer_api/run.py`.

### Defaults

| Variable | Default | Notes |
|----------|---------|-------|
| `APP_HOST` | `127.0.0.1` | |
| `APP_PORT` | `8000` | |
| `APP_RELOAD` | `false` | Enables uvicorn auto-reload |
| `TOPICER_API_CONFIGS_DIR` | `./configs` | Directory scanned for topicer YAMLs |
| `TOPICER_API_CONFIGS_EXTENSION` | – | File extension filter for configs |

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/configs` | List loaded topicer configs |
| POST | `/v1/topics/discover/texts/sparse` | Sparse topic discovery on supplied texts |
| POST | `/v1/topics/discover/texts/dense`  | Dense topic discovery on supplied texts |
| POST | `/v1/topics/discover/db/sparse`    | Sparse topic discovery on DB-stored chunks |
| POST | `/v1/topics/discover/db/dense`     | Dense topic discovery on DB-stored chunks |
| POST | `/v1/tags/propose/texts`           | Tag proposal on a supplied `TextChunk` |
| POST | `/v1/tags/propose/db`              | Tag proposal on DB-stored chunks (single response) |
| POST | `/v1/tags/propose/db/stream`       | Tag proposal on DB-stored chunks, **streaming NDJSON** |

#### `POST /v1/tags/propose/db/stream`

* Content type: `application/x-ndjson`
* Each line is a JSON-serialized `TextChunkWithTagSpanProposals` object,
  emitted as soon as the corresponding LLM call resolves.
* Returns HTTP 404 if `config_name` is unknown.
* Returns HTTP 409 if the loaded topicer does not implement
  `propose_tags_in_db_stream` (currently only `LLMTopicer` does).

Example client (Python / `httpx`):

```python
import httpx, json

with httpx.stream("POST",
                 "http://localhost:8000/v1/tags/propose/db/stream",
                 params={"config_name": "llm_default"},
                 json={"tag": tag.model_dump(mode="json"),
                       "db_request": {"collection_id": str(coll_id)}}) as r:
    for line in r.iter_lines():
        if line:
            yield json.loads(line)
```

Swagger UI: `http://localhost:8000/docs`.

---

## 7. Configuration files

Per-method example configurations live under:

* `configs/` – defaults bundled with the source tree
* `deploy/data/configs/` – configurations baked into the Docker image
* `examples/*/config.yaml` – minimal per-feature examples

The previous root-level `config.yaml` has been removed; pick (or copy) one of the
files above when running the API or Python factory.

---

## 8. Migration notes (from `main`)

If you are upgrading existing client code from `main` to this branch:

1. **`Tag.examples`** – change `list[TextWithSpan]` to `list[str]`. Pass the
   example phrases directly; remove any span/offset bookkeeping.
2. **`DBRequest`** – the new `document_id` field is optional; existing payloads
   continue to work unchanged.
3. **Streaming endpoint** – consider switching long-running tagging requests to
   `POST /v1/tags/propose/db/stream` for better UX.
4. **Root `config.yaml`** – any tooling that referenced the file at the repo
   root must be repointed to `configs/config.yaml` (or another location set via
   `TOPICER_API_CONFIGS_DIR`).
5. **Custom `LLMTagProposal` consumers** – the private schema in
   `topicer/tagging/tagging_schemas.py` now uses `tag_id: UUID` instead of an
   embedded `Tag`. Resolution to a full `Tag` happens inside `LLMTopicer`, so
   the public `TagSpanProposal` API is unchanged.
