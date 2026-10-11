"""OpenRouter chat client: one stateless JSON-mode call with retries."""

import json
import os
import time
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

from promptgen import config

URL = "https://openrouter.ai/api/v1/chat/completions"
load_dotenv()


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    data: dict
    prompt_tokens: int
    completion_tokens: int
    model: str


def default_model() -> str:
    return os.getenv("OPENROUTER_MODEL") or config.DEFAULT_MODEL


def _parse(content: str) -> dict:
    s = content.strip()
    if s.startswith("```"):  # some models fence JSON even in json_object mode
        s = s.split("\n", 1)[1].rsplit("```", 1)[0]
    data = json.loads(s)
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    return data


def chat_json(system: str, user: str, model: str | None = None,
              client: httpx.Client | None = None) -> LLMResult:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise LLMError("OPENROUTER_API_KEY is not set (see .env.example)")
    model = model or default_model()
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": config.TEMPERATURE,
        "response_format": {"type": "json_object"},
        "reasoning": {"enabled": False},  # thinking tokens made batches ~6x slower for no visible gain
    }
    headers = {"Authorization": f"Bearer {key}", "X-OpenRouter-Title": "Patient Prompt Agent"}
    client = client or httpx.Client(timeout=config.TIMEOUT_S)

    last = None
    for attempt in range(config.RETRIES + 1):
        if attempt:
            time.sleep(2 ** (attempt - 1))  # 1, 2, 4 s
        try:
            r = client.post(URL, json=body, headers=headers)
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                continue
            if r.status_code >= 400:  # bad key, bad model: retrying won't help
                raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
            resp = r.json()
            if "error" in resp:  # OpenRouter can return 200 with an upstream error
                last = f"upstream error: {resp['error']}"
                continue
            usage = resp.get("usage") or {}
            return LLMResult(
                data=_parse(resp["choices"][0]["message"]["content"] or ""),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                model=resp.get("model", model),
            )
        except (httpx.TransportError, ValueError, KeyError, IndexError) as e:
            last = f"{type(e).__name__}: {e}"
    raise LLMError(f"gave up after {config.RETRIES + 1} attempts; last: {last}")


if __name__ == "__main__":
    t = time.time()
    res = chat_json(
        'You write questions a patient would type into ChatGPT. Reply with JSON: '
        '{"prompts":[{"text":"...","intent":"...","competitor":null}]}',
        "Write 3 first-person questions about the psoriasis drug Brandex. "
        f"intent must be one of {config.INTENTS}.",
    )
    print(json.dumps(res.data, indent=2))
    print(f"model={res.model} tokens in/out={res.prompt_tokens}/{res.completion_tokens} "
          f"time={time.time() - t:.1f}s")
