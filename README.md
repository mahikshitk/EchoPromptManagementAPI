# EchoPromptManagementAPI — Prompt Agent

Generates the questions people type into ChatGPT, Gemini or Perplexity about a pharma brand, for every
audience × topic pair in the payload (patient, HCP, policy & payer, media / investor × 7 topics).
Each pair gets `promptsPerPair` prompts, capped by `maxPrompts`; when the cap bites, a random set of pairs
gets one fewer. Angles rotate randomly each run. Patient prompts use real Google Autocomplete / Trends
questions as few-shot style references. Generation via OpenRouter.

## Setup

```
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt pytest   # pytest is dev-only, not shipped
cp .env.example .env   # set OPENROUTER_API_KEY, optionally OPENROUTER_MODEL
```

## Run

```
python -m promptgen.generator                          # -> output/<requestId>.json
python -m promptgen.generator --dry-run                # print the LLM prompts per pair, no calls
python -m promptgen.generator --n 100 --per-pair 4     # override maxPrompts / promptsPerPair
python -m promptgen.generator other_payload.json --no-sources --seed 1
python -m promptgen.generator --offline                # Autocomplete/Trends from the store only, no fetching
```

## API

```
uvicorn api:app --reload                                # local server on :8000
curl localhost:8000/health
curl -X POST localhost:8000/generate -H 'content-type: application/json' -d @sample_payload.json
```

`POST /generate` takes the input payload and returns the output JSON (422 on an invalid payload).
Run warnings (unknown codes, short pairs, failed LLM calls) are logged, not returned, since the
output schema is fixed. `api.handler` is the Lambda entry point (Mangum).

Lambda notes: set `OPENROUTER_API_KEY` (and optionally `OPENROUTER_MODEL`), `HARVEST_DB_PATH=/tmp/harvest.sqlite`
(only `/tmp` is writable) and `HARVEST_DB_S3_URI` so the harvest store persists across containers; the role
needs `s3:GetObject`/`s3:PutObject` on that key. A run takes ~7-25s, so allow a timeout of at least 60s.

## Harvest store

Every Autocomplete and Trends result ever fetched is kept in an append-only SQLite file: a refresh
(after 7 days) adds new questions and never removes old ones, and when Google rate-limits or fails,
everything stored for that brand / disease is still used. Set `HARVEST_DB_S3_URI=s3://bucket/key` to
keep the file in S3: it is merged in at the start of a run and uploaded (after re-merging) when new
results were fetched. S3 errors are logged and never fail a run.

## Tests

```
pytest -q                                              # offline unit tests
RUN_LIVE=1 pytest tests/test_acceptance_live.py -v -s  # acceptance against the real model (a few minutes)
```

## Layout

```
api.py                    FastAPI app (POST /generate, GET /health) and Lambda handler
promptgen/config.py       audiences, topics, intents, angles, rules, limits (prompt content lives here)
promptgen/schema.py       Pydantic input/output models, pair allocation, shared CLI
promptgen/harvest.py      Autocomplete, Trends, append-only SQLite store (7-day refresh), optional S3 sync
promptgen/fewshot.py      relevance filter, topic tagging, fallback top-up, rotation sampler
promptgen/llm.py          OpenRouter client (JSON mode, retries, reasoning off)
promptgen/generator.py    prompt building, parallel per-pair batch loop, top-up
promptgen/postprocess.py  per-audience validation, dedup, groundingUsed, output assembly
```
