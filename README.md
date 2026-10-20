# EchoPromptManagementAPI — Prompt Agent

Generates the questions people type into ChatGPT, Gemini or Perplexity about a pharma brand, for every
audience × topic pair in the payload (patient, HCP, policy & payer, media / investor × 7 topics).
Each pair gets `promptsPerPair` prompts, capped by `maxPrompts`; when the cap bites, a random set of pairs
gets one fewer. Angles rotate randomly each run. Patient prompts use real Google Autocomplete / Trends
questions as few-shot style references. Generation via OpenRouter.

## Setup

```
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env   # set OPENROUTER_API_KEY, optionally OPENROUTER_MODEL
```

## Run

```
python -m promptgen.generator                          # -> output/<requestId>.json
python -m promptgen.generator --dry-run                # print the LLM prompts per pair, no calls
python -m promptgen.generator --n 100 --per-pair 4     # override maxPrompts / promptsPerPair
python -m promptgen.generator --brand Tylenol --disease Headache --competitor Advil
python -m promptgen.generator --stage Pre-launch --no-sources --seed 1
python -m promptgen.generator --offline                # Autocomplete/Trends from the store only, no fetching
```

## Harvest store

Every Autocomplete and Trends result ever fetched is kept in an append-only SQLite file: a refresh
(after 7 days) adds new questions and never removes old ones, and when Google rate-limits or fails,
everything stored for that brand / disease is still used. Set `HARVEST_DB_S3_URI=s3://bucket/key` to
keep the file in S3: it is merged in at the start of a run and uploaded (after re-merging) when new
results were fetched. S3 errors are logged and never fail a run.

Each module also runs standalone for inspection: `promptgen.schema` (pairs and allocation),
`promptgen.llm` (one smoke call), `promptgen.harvest`, `promptgen.fewshot`, `promptgen.postprocess`.

## Tests

```
pytest -q                                              # offline unit tests
RUN_LIVE=1 pytest tests/test_acceptance_live.py -v -s  # acceptance against the real model (a few minutes)
```

## Layout

```
promptgen/config.py       audiences, topics, intents, angles, rules, limits (prompt content lives here)
promptgen/schema.py       Pydantic input/output models, pair allocation, shared CLI
promptgen/harvest.py      Autocomplete, Trends, append-only SQLite store (7-day refresh), optional S3 sync
promptgen/fewshot.py      relevance filter, topic tagging, fallback top-up, rotation sampler
promptgen/llm.py          OpenRouter client (JSON mode, retries, reasoning off)
promptgen/generator.py    prompt building, parallel per-pair batch loop, top-up
promptgen/postprocess.py  per-audience validation, dedup, groundingUsed, output assembly
```
