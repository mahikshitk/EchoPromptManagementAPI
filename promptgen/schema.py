"""Input/output models. Field names are camelCase to match the client schema exactly."""

import argparse
import json

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from promptgen import config


class _In(BaseModel):
    model_config = ConfigDict(extra="allow")


class Market(_In):
    name: str = ""
    isoCode: str = "GB"


class Grounding(_In):
    brand: str
    market: Market = Market()
    stage: str = "Launched"
    marketAccess: str = ""
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
    id: str = ""
    text: str
    category: str = ""
    audience: str = ""


class Options(_In):
    maxPrompts: int = Field(47, ge=1, le=100)
    promptsPerPair: int = Field(2, ge=0)
    includeCompetitorPrompts: bool = True
    includeGoogleTrends: bool = True
    includeAutocomplete: bool = True
    includeReddit: bool = False
    language: str = "en-GB"


class Payload(_In):
    requestId: str
    runId: str
    grounding: Grounding
    audiences: list[Audience]
    topics: list[Topic]
    existingPrompts: list[ExistingPrompt] = []
    options: Options = Options()

    @model_validator(mode="after")
    def _check_scope(self):
        if not any(a.code == config.AUDIENCE for a in self.audiences):
            raise ValueError(f"payload has no '{config.AUDIENCE}' audience")
        if not self.enabled_topics():
            raise ValueError(f"payload has none of the topics {list(config.TOPIC_WEIGHTS)}")
        return self

    @property
    def audience(self) -> Audience:
        return next(a for a in self.audiences if a.code == config.AUDIENCE)

    def enabled_topics(self) -> list[Topic]:
        """In-scope topics, in config order, with name/categoryId from the payload."""
        by_code = {t.code: t for t in self.topics}
        return [by_code[c] for c in config.TOPIC_WEIGHTS if c in by_code]

    def with_grounding(self, brand: str | None = None, disease_areas: list[str] | None = None,
                       competitors: list[str] | None = None) -> "Payload":
        """Copy with user overrides (UI/CLI). None keeps the payload value."""
        clean = lambda xs: [x.strip() for x in xs if x.strip()]
        update = {}
        if brand is not None:
            if not brand.strip():
                raise ValueError("brand must not be empty")
            update["brand"] = brand.strip()
        if disease_areas is not None:
            update["diseaseAreas"] = clean(disease_areas)
        if competitors is not None:
            update["competitors"] = clean(competitors)
        return self.model_copy(update={"grounding": self.grounding.model_copy(update=update)})

    def do_not_repeat(self) -> list[str]:
        return [p.text for p in self.existingPrompts if p.audience in ("", config.AUDIENCE)]

    def warnings(self) -> list[str]:
        w = []
        ignored_aud = [a.code for a in self.audiences if a.code != config.AUDIENCE]
        if ignored_aud:
            w.append(f"Ignored audiences (v1 is patient only): {', '.join(ignored_aud)}")
        enabled = {t.code for t in self.enabled_topics()}
        ignored_top = [t.code for t in self.topics if t.code not in enabled]
        if ignored_top:
            w.append(f"Ignored topics (out of scope): {', '.join(ignored_top)}")
        missing = [c for c in config.TOPIC_WEIGHTS if c not in enabled]
        if missing:
            w.append(f"Topics missing from payload, skipped: {', '.join(missing)}")
        if self.options.includeReddit:
            w.append("includeReddit is ignored in v1")
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
    """Shared CLI: payload path plus brand/disease/competitor overrides."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("payload", nargs="?", default="sample_payload.json")
    ap.add_argument("--brand")
    ap.add_argument("--disease", action="append", help="repeatable; replaces payload diseaseAreas")
    ap.add_argument("--competitor", action="append", help="repeatable; replaces payload competitors")
    if extra_args:
        extra_args(ap)
    a = ap.parse_args()
    p = Payload.model_validate_json(open(a.payload).read())
    return p.with_grounding(a.brand, a.disease, a.competitor), a


if __name__ == "__main__":
    p, _ = load_cli("Parse a payload and show what v1 will use")
    print(json.dumps({
        "brand": p.grounding.brand,
        "diseaseAreas": p.grounding.diseaseAreas,
        "competitors": p.grounding.competitors,
        "stage": p.grounding.stage,
        "market": p.grounding.market.isoCode,
        "audience": p.audience.model_dump(),
        "topics": [t.model_dump() for t in p.enabled_topics()],
        "doNotRepeat": p.do_not_repeat(),
        "options": p.options.model_dump(),
    }, indent=2))
    for w in p.warnings():
        print("WARNING:", w)
