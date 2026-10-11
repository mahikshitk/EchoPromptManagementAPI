import json

import httpx
import pytest

from promptgen import llm


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)


def ok(content: str) -> httpx.Response:
    return httpx.Response(200, json={
        "model": "m", "usage": {"prompt_tokens": 5, "completion_tokens": 7},
        "choices": [{"message": {"content": content}}],
    })


def client(*responses):
    it = iter(responses)
    calls = []

    def handler(req):
        calls.append(json.loads(req.content))
        return next(it)
    c = httpx.Client(transport=httpx.MockTransport(handler))
    c.calls = calls
    return c


def test_success_and_request_shape():
    c = client(ok('{"prompts": []}'))
    r = llm.chat_json("sys", "user", model="x/y", client=c)
    assert r.data == {"prompts": []} and (r.prompt_tokens, r.completion_tokens) == (5, 7)
    body = c.calls[0]
    assert body["model"] == "x/y" and body["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_strips_code_fence():
    r = llm.chat_json("s", "u", client=client(ok('```json\n{"a": 1}\n```')))
    assert r.data == {"a": 1}


def test_retries_429_5xx_and_bad_json():
    c = client(httpx.Response(429), httpx.Response(503), ok("not json"), ok('{"a": 1}'))
    assert llm.chat_json("s", "u", client=c).data == {"a": 1}
    assert len(c.calls) == 4


def test_gives_up_after_retries():
    c = client(*[httpx.Response(500)] * 4)
    with pytest.raises(llm.LLMError, match="gave up"):
        llm.chat_json("s", "u", client=c)


def test_401_fails_fast():
    c = client(httpx.Response(401, text="bad key"))
    with pytest.raises(llm.LLMError, match="401"):
        llm.chat_json("s", "u", client=c)
    assert len(c.calls) == 1


def test_missing_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(llm.LLMError, match="not set"):
        llm.chat_json("s", "u")
