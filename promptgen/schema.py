"""Input/output models. Field names are camelCase to match the client schema exactly."""

import argparse
import random

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from promptgen import config


class _In(BaseModel):
    model_config = ConfigDict(extra="allow")


class Market(_In):
    name: str = ""
    isoCode: str = "GB"


class Websites(_In):
    product: str = ""
    corporate: str = ""
    other: list[str] = []


class Grounding(_In):
    brand: str
    market: Market = Market()
    stage: str = "Launched"
    marketAccess: str = ""
    websites: Websites = Websites()
    diseaseAreas: list[str] = []
    competitors: list[str] = []


class Audience(_In):
    code: str
    name: str
    voice: str = ""


class Topic(_In):
    code: str
    name: str
    categoryId: int


class ExistingPrompt(_In):
    text: str
    audience: str = ""


class Options(_In):
    maxPrompts: int = Field(47, ge=1, le=100)
    promptsPerPair: int = Field(2, ge=1)
    includeCompetitorPrompts: bool = True
    includeGoogleTrends: bool = True
    includeAutocomplete: bool = True
    includeReddit: bool = False
    language: str = "en-GB"


def _known(items: list, codes: list[str]) -> list:
    """Items whose code is known, in payload order, first occurrence of each code."""
    out = {}
    for x in items:
        if x.code in codes:
            out.setdefault(x.code, x)
    return list(out.values())


class Payload(_In):
    requestId: str
    runId: str
    grounding: Grounding
    audiences: list[Audience]
    topics: list[Topic]
    existingPrompts: list[ExistingPrompt] = []
    benchmark: dict | None = None  # accepted, not used yet; documents become grounding later
    options: Options = Options()

    @model_validator(mode="after")
    def _check_scope(self):
        if not self.active_audiences():
            raise ValueError(f"payload has none of the audiences {config.AUDIENCES}")
        if not self.active_topics():
            raise ValueError(f"payload has none of the topics {config.TOPICS}")
        return self

    def active_audiences(self) -> list[Audience]:
        return _known(self.audiences, config.AUDIENCES)

    def active_topics(self) -> list[Topic]:
        return _known(self.topics, config.TOPICS)

    def pairs(self) -> list[tuple[Audience, Topic]]:
        return [(a, t) for a in self.active_audiences() for t in self.active_topics()]

    def allocate(self, seed: int | None = None) -> dict[tuple[str, str], int]:
        """(audience code, topic code) -> prompt count. Every pair targets promptsPerPair; if that
        exceeds maxPrompts, the total is spread evenly and a random set of pairs gets the extra one,
        so which pairs come up short changes from run to run."""
        keys = [(a.code, t.code) for a, t in self.pairs()]
        total = min(self.options.maxPrompts, len(keys) * self.options.promptsPerPair)
        base, extra = divmod(total, len(keys))
        lucky = set(random.Random(seed).sample(keys, extra))
        return {k: base + (k in lucky) for k in keys}

    def do_not_repeat(self, audience: str) -> list[str]:
        """existingPrompts for this audience; an empty audience applies to all."""
        return [p.text for p in self.existingPrompts if p.audience in ("", audience)]

    def warnings(self) -> list[str]:
        w = []
        unknown_aud = [a.code for a in self.audiences if a.code not in config.AUDIENCES]
        if unknown_aud:
            w.append(f"Ignored unknown audiences: {', '.join(unknown_aud)}")
        unknown_top = [t.code for t in self.topics if t.code not in config.TOPICS]
        if unknown_top:
            w.append(f"Ignored unknown topics: {', '.join(unknown_top)}")
        n, ppp = len(self.pairs()), self.options.promptsPerPair
        if n * ppp > self.options.maxPrompts:
            w.append(f"maxPrompts {self.options.maxPrompts} caps {n} pairs x {ppp} = {n * ppp}; "
                     "some pairs get fewer" + (" (some get none)" if self.options.maxPrompts < n else ""))
        if self.options.includeReddit:
            w.append("includeReddit is ignored in this version")
        if (self.benchmark or {}).get("documents"):
            w.append("benchmark documents are ignored in this version")
        return w


# ---- Output: exact client schema, no extras ----

class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OutAudience(_Out):
    code: str
    name: str


class OutTopic(_Out):
    code: str
    name: str
    categoryId: int


class OutPrompt(_Out):
    tempId: str = Field(pattern=r"^g_\d{3}$")
    text: str
    audience: OutAudience
    topic: OutTopic
    competitor: str | None
    intent: str
    groundingUsed: list[str]

    @field_validator("intent")
    @classmethod
    def _intent(cls, v: str) -> str:
        if v not in config.INTENTS:
            raise ValueError(f"unknown intent {v!r}")
        return v


class Output(_Out):
    requestId: str
    runId: str
    generatedAt: str
    prompts: list[OutPrompt]


def load_cli(description: str = "", extra_args=None) -> tuple[Payload, argparse.Namespace]:
    """Shared CLI: payload path plus any extra arguments."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("payload", nargs="?", default="sample_payload.json")
    if extra_args:
        extra_args(ap)
    a = ap.parse_args()
    return Payload.model_validate_json(open(a.payload).read()), a
