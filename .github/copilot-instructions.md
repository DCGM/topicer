# Copilot / AI Agent Instructions for `topicer`

Purpose: Help an AI coding agent be immediately productive editing and extending this small Python project.

Quick summary
- **Project role:** a minimal tag-proposal module that models tag objects and provides async hooks to propose tags for text chunks (`tag_proposals.py`).
- **Key files:** `tag_proposals.py`, `config.yaml`, `requirements.txt`, `README.md`.

Big picture
- `tag_proposals.py` declares the domain models (Pydantic `BaseModel` types) and a `TagProposal` class that reads configuration and exposes two async hooks: `propose_tags(text, tag)` and `propose_tags_in_db(tag, db_request)`.
- External integrations are expected but not implemented: `weaviate` (vector DB) and `openai` (model API). Configs live in `config.yaml`.
- Design choices: heavy use of Pydantic models and strong typing (UUIDs, typed lists, union `|` syntax) — assume Python >= 3.10 and Pydantic v2.

What to change and where (common tasks)
- Add a new proposal implementation: edit `TagProposal` in `tag_proposals.py` — implement `propose_tags` to return `TextChunkWithTags` constructed from input chunk.
- Add DB-backed proposal logic: implement `propose_tags_in_db` to query whatever DB client you wire (Weaviate) and return a `list[TextChunkWithTags]`.
- Add tests: create a `tests/` folder and unit tests that instantiate Pydantic models and call the async methods via `pytest` + `pytest-asyncio`.

Developer workflows and commands
- Create virtual env and install deps:
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```
- Run the module (prints config):
```
python tag_proposals.py
```
- Linting/formatting: no project config present — follow repository style (keep simple, black/ruff optional).

Project-specific conventions & patterns
- Use Pydantic v2 style models throughout (`class X(BaseModel): ...`). Return and accept Pydantic objects (e.g., `TextChunk`, `TextChunkWithTags`).
- Use `UUID` for ids (model fields expect `uuid.UUID`). Keep `examples` as `list[TextWithSpan]` when present.
- Async API surface: methods intended to be awaited. Keep signatures as declared to preserve compatibility with potential async DB/HTTP calls.
- `Config` in `tag_proposals.py` reads `config.yaml` via `yaml.safe_load` and prints it — treat it as canonical config source for local dev.

Integration points & environment
- `config.yaml` keys to populate before calling remote systems:
  - `weaviate.host`, `weaviate.port` — used if you implement Weaviate client code
  - `openai.api_key`, `openai.model` — used if you implement calls to OpenAI
- Secrets are stored in `config.yaml` in this repo; prefer environment variables or a secrets manager in production.

Examples (concrete references)
- Model: `Tag` (in `tag_proposals.py`) — fields: `id: UUID`, `name: str`, `description: str | None`, `examples: list[TextWithSpan] | None`.
- Hook: `async def propose_tags(self, text: TextChunk, tag: list[Tag]) -> TextChunkWithTags` — return the same `id`/`text` and a `tags` list of `TagSpanProposal`.
- Config: `config.yaml` contains `weaviate` and `openai` blocks; code instantiates `TagProposal("config.yaml")` in `__main__`.

Agent behavior guidelines (specific, actionable)
- Preserve type annotations and Pydantic models; prefer returning model instances rather than raw dicts.
- If adding new dependencies, update `requirements.txt` and mention why in the PR description.
- Keep changes minimal and focused: avoid refactors that alter public model fields or method signatures unless necessary.
- When adding remote calls, add a clear fallback/mock implementation so local runs (and CI) don't require external services.

Questions for maintainers
- Confirm preferred Python minor version (>=3.10 assumed). If different, update instructions.
- Confirm whether secrets should remain in `config.yaml` or be moved to env vars/secrets manager.

If something in these notes is unclear, ask for one-line examples you want the agent to implement (e.g., "implement basic OpenAI-based tag proposal using `openai` key").
