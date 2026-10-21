# EchoPromptManagementAPI — Prompt Agent

Generates the questions people type into ChatGPT, Gemini or Perplexity about a pharma brand, for every
audience × topic pair in the payload (patient, HCP, policy & payer, media / investor × 7 topics).
Each pair gets `promptsPerPair` prompts, capped by `maxPrompts`; when the cap bites, a random set of pairs
gets one fewer. Angles rotate randomly each run. Patient prompts use real Google Autocomplete / Trends
questions as few-shot style references. Generation via OpenRouter.

## Setup

```
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements-dev.txt   # runtime deps + boto3, uvicorn, pytest (not shipped)
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

The UI calls the API with the same Cognito access token it sends to RunManagementAPI
(`Authorization: Bearer <token>`). Generation runs in the background:

1. `POST /generate` with the input payload validates it, queues a generation and returns **202**
   `{"generationId", "requestId", "runId", "status": "queued", "statusUrl"}` with a `Location` header.
2. `GET /generate/{generationId}` returns `status` (`queued` | `running` | `completed` | `failed`),
   `progress` `{"pairsDone", "pairsTotal"}`, timestamps, and `result` (the output JSON) once `completed`
   (`null` until then). Poll it every ~2s; a run takes ~10-60s.

| Status | When |
|---|---|
| 401 | missing or invalid token, an ID token, or a token from a client other than the UI clients |
| 403 | the token lacks `run-management/run:management` |
| 404 | unknown generation, or another user's (the same response, so ids cannot be probed) |
| 422 | invalid payload or generation id |
| 429 | the user already has 2 queued or running generations |
| 503 | the worker could not be started (the generation is marked failed) |

A failed or timed-out generation (no progress for 15 min) reports `status: failed` with a generic `error`;
the cause is in the worker's logs and the table's `error` column. Run warnings (unknown codes, short pairs,
failed LLM calls) are logged, not returned, since the output schema is fixed. `GET /health` needs no token.

Locally, set `AUTH_DISABLED=1` (never in AWS): jobs live in memory and run on a background thread.

```
AUTH_DISABLED=1 uvicorn api:app --reload                # local server on :8000
curl localhost:8000/health
curl -X POST localhost:8000/generate -H 'content-type: application/json' -d @sample_payload.json
curl localhost:8000/generate/<generationId>
```

Set `DATABASE_URL` to a PostgreSQL with the prerequisite tables to keep jobs and the harvest store there
instead of in memory and the local SQLite file.

## Authentication

The same rules as RunManagementAPI, checked twice. API Gateway's JWT authorizer accepts only tokens issued
by the RunManagement Cognito user pool for the interactive or admin client. The app then checks again
([promptgen/auth.py](promptgen/auth.py)): signature and issuer, `token_use` is `access`, `client_id` is one of
`AUTH_ALLOWED_CLIENT_IDS`, and the `run-management/run:management` scope is present. If `AUTH_AUTHORITY` or
`AUTH_ALLOWED_CLIENT_IDS` is unset the API returns 503 rather than running unauthenticated. Each generation
belongs to the token's `sub`.

## Deployment (AWS)

Three Lambdas from one container image ([Dockerfile](Dockerfile)), deployed by the CDK stack in
[infra/](infra/stack.py) to the same account, VPC and Cognito user pool as RunManagementAPI:

| Lambda | Handler | Network | Does |
|---|---|---|---|
| `prompt-management-<env>-api` | `api.handler` | outside VPC | HTTP API behind the JWT authorizer |
| `prompt-management-<env>-worker` | `worker.handler` | outside VPC | runs a generation (OpenRouter, Google), 10 min timeout, no retries |
| `prompt-management-<env>-database` | `database.handler` | private subnets | the only code that connects to AtrionDB (jobs and harvest store) |

As in RunManagementAPI, the API and worker invoke the database Lambda and never hold PostgreSQL
credentials; it reads the existing AtrionDB secret (`ATRION_DB_SECRET_ARN`, RDS shape or
`ConnectionStrings.AtrionDb`) and enforces TLS. The worker stays outside the VPC so it needs no NAT route.

Before the first deploy:

1. A database owner applies [database/prerequisites/](database/prerequisites/) `001` and `002` to AtrionDB in order,
   and grants the secret's application role `SELECT, INSERT, UPDATE` on `public.atrion_prompt_generation`,
   `public.atrion_prompt_harvest_fetch` and `public.atrion_prompt_harvest_result`.
2. Create a Secrets Manager secret whose whole plaintext value is the OpenRouter API key.
3. Take `userPoolId`, `interactiveClientId` and `adminClientId` from the RunManagement stack outputs, and reuse
   its VPC, subnet, security-group and AtrionDB secret values.

Deploy (Docker must be running to build the image):

```
uv pip install -r infra/requirements.txt
npm ci                                                 # the pinned CDK CLI
npx cdk deploy PromptManagement-<env> \
  -c environment=<env> -c account=<account> -c region=<region> \
  -c vpcId=<vpc> -c privateSubnetIds=<subnet-a,subnet-b> -c privateSubnetAvailabilityZones=<az-a,az-b> \
  -c databaseLambdaSecurityGroupId=<sg> -c atrionDbSecretArn=<complete ARN> \
  -c openRouterSecretArn=<complete ARN> \
  -c userPoolId=<pool> -c interactiveClientId=<id> -c adminClientId=<id> \
  -c corsAllowedOrigins=<optional; https://ui.example.com, every origin when unset> \
  -c openRouterModel=<optional model id>
```

The `ApiUrl` output is the UI's base URL.

Normally, deploy from GitHub Actions instead: **Actions > Deploy > Run workflow** with the target environment
(OIDC role, no stored AWS keys). The one-time AWS and GitHub environment setup is in
[docs/operations/github-actions-deployment.md](docs/operations/github-actions-deployment.md).

## Harvest store

Every Autocomplete and Trends result ever fetched is kept, append-only: a refresh (after 7 days) adds new
questions and never removes old ones, and when Google rate-limits or fails, everything stored for that
brand / disease is still used. On AWS it lives in AtrionDB (`atrion_prompt_harvest_fetch` and
`atrion_prompt_harvest_result`, through the database Lambda), so concurrent workers never lose each other's
rows. Locally it is a SQLite file at `HARVEST_DB_PATH` (default `.cache/harvest.sqlite`), or AtrionDB tables
when `DATABASE_URL` is set. A run makes one batched read before fetching and one batched write after; store
errors are logged and never fail a run.

## Tests

```
pytest -q                                              # offline unit tests
RUN_LIVE=1 pytest tests/test_acceptance_live.py -v -s  # acceptance against the real model (a few minutes)
TEST_DATABASE_URL=postgresql://... pytest tests/test_jobs.py tests/test_harvest.py  # stores against a scratch PostgreSQL (drops and recreates the tables)
```

## Layout

```
api.py                    FastAPI app (POST /generate, GET /generate/{id}, GET /health) and Lambda handler
worker.py                 runs one generation job; worker Lambda handler and local thread runner
database.py               AtrionDB Lambda handler (in the VPC) serving the job-store operations
promptgen/auth.py         Cognito access-token and scope check (RunManagementAPI's rules)
promptgen/db.py           AtrionDB connection (secret, TLS) and the database-Lambda proxy
promptgen/jobs.py         generation job store: PostgreSQL, database-Lambda proxy, in-memory
promptgen/runtime.py      logging and Secrets Manager helpers for the Lambdas
database/prerequisites/   AtrionDB scripts a database owner applies before deploying
infra/                    CDK stack (Python): Lambdas, HTTP API + JWT authorizer
promptgen/config.py       audiences, topics, intents, angles, rules, limits (prompt content lives here)
promptgen/schema.py       Pydantic input/output models, pair allocation, shared CLI
promptgen/harvest.py      Autocomplete, Trends, append-only store (7-day refresh): AtrionDB or local SQLite
promptgen/fewshot.py      relevance filter, topic tagging, fallback top-up, rotation sampler
promptgen/llm.py          OpenRouter client (JSON mode, retries, reasoning off)
promptgen/generator.py    prompt building, parallel per-pair batch loop, top-up
promptgen/postprocess.py  per-audience validation, dedup, groundingUsed, output assembly
```
