# EchoPromptManagementAPI — Patient Prompt Agent (v1 prototype)

Generates `n` patient-voice prompts about a pharma brand (the questions a patient would type into
ChatGPT, Gemini or Perplexity). Real Google Autocomplete / Trends questions are used as few-shot
style references. Generation via OpenRouter. Scope: audience `patient`; topics `safety`,
`efficacy`, `access_cost`.

## Setup

```
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env   # set OPENROUTER_API_KEY, optionally OPENROUTER_MODEL
```

## Run

```
streamlit run app.py                                   # demo UI
python -m promptgen.generator                          # CLI, n from payload -> output/<requestId>.json
python -m promptgen.generator --n 100 --brand Tylenol --disease Headache --competitor Advil
python -m promptgen.generator --stage Pre-launch --no-sources
```

Each module also runs standalone for inspection: `promptgen.schema`, `promptgen.llm` (one smoke
call), `promptgen.harvest`, `promptgen.fewshot`, `promptgen.postprocess`.

## Tests

```
pytest -q                                              # offline unit tests
RUN_LIVE=1 pytest tests/test_acceptance_live.py -v -s  # design §7 acceptance, real model (~1 min)
```

## Layout

```
app.py                    Streamlit UI
promptgen/config.py       weights, intents, angles, keyword map, fallback examples, limits
promptgen/schema.py       Pydantic input/output models, grounding overrides, shared CLI
promptgen/harvest.py      Autocomplete, Trends, SQLite cache (.cache/, 7-day TTL, stale on error)
promptgen/fewshot.py      relevance filter, topic tagging, fallback top-up, rotation sampler
promptgen/llm.py          OpenRouter client (JSON mode, retries, reasoning off)
promptgen/generator.py    allocation, parallel per-topic batch loop, top-up
promptgen/postprocess.py  validation, dedup, groundingUsed, output assembly
```
