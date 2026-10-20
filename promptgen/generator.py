"""Prompt building, the per-pair batch loop, top-up, and the end-to-end run."""

import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from promptgen import config, llm
from promptgen.fewshot import Sampler, build_pool
from promptgen.harvest import HarvestResult, harvest
from promptgen.postprocess import Deduper, assemble, filter_batch
from promptgen.schema import Audience, Output, Payload, Topic

# ---- angles ----


def batch_sizes(q: int) -> list[int]:
    """ceil(q/BATCH_SIZE) near-equal batches, e.g. 19 -> [10, 9]."""
    if q <= 0:
        return []
    k = math.ceil(q / config.BATCH_SIZE)
    return [q // k + (1 if i < q % k else 0) for i in range(k)]


def angles_for(topic: str, audience: str, p: Payload) -> list[str]:
    """Angles for a pair, "vs competitor" expanded per competitor or dropped if comparisons are off."""
    out = []
    for a in config.ANGLES[topic][audience]:
        if a != config.VS_COMPETITOR:
            out.append(a)
        elif p.options.includeCompetitorPrompts:
            out += [f"vs {c}" for c in p.grounding.competitors]
    return out


def angle_orders(p: Payload, alloc: dict, seed: int | None = None) -> dict[tuple[str, str], list[str]]:
    """Each allocated pair's angles in a random order: batches take angles in this order, so every
    run starts each pair on a different angle."""
    rng = random.Random(seed)
    out = {}
    for a, t in p.pairs():
        if alloc[a.code, t.code]:
            angles = angles_for(t.code, a.code, p)
            out[a.code, t.code] = rng.sample(angles, len(angles))
    return out


# ---- prompts ----

SYSTEM = """You write the questions that {persona} would type into ChatGPT, Gemini or Perplexity about a medicine.

Audience: {audience}. Voice: {voice}.
Market: {market}. Language: {language}. Use the healthcare-system terms, institutions and spelling natural for that market.

Rules:
- Write questions only, never answers. Every question ends with "?".
{audience_rules}
- Do not invent facts: no made-up numbers, trial results, prices, dates, approvals, events or company names.
- Comparisons and criticisms are questions, never claims.
{stage_rule}- Never repeat or paraphrase anything in the DO NOT REPEAT list. Every question must be distinct from the others.
- Vary the phrasing and length: mix short searches with fuller, specific questions.
{style_rule}
Intents (pick one per question):
{intents}

Reply with JSON only:
{{"prompts": [{{"text": "...", "intent": "<intent code>", "competitor": "<exact competitor name or null>"}}]}}"""


def system_prompt(p: Payload, audience: Audience) -> str:
    g = p.grounding
    prof = config.AUDIENCE_PROFILES[audience.code]
    stage_rule = "" if g.stage.lower() == "launched" else (
        f"- {g.brand} is not launched yet (stage: {g.stage}). Never imply it is approved, available, "
        "prescribed, or that anyone is already taking it.\n")
    style_rule = ("- Match the phrasing style and length of the EXAMPLE SEARCHES, but never copy them.\n"
                  if prof["fewshot"] else "")
    return SYSTEM.format(
        persona=prof["persona"], audience=audience.name, voice=audience.voice or "natural for this audience",
        market=g.market.name or g.market.isoCode, language=p.options.language,
        audience_rules="\n".join(f"- {r}" for r in prof["rules"]), stage_rule=stage_rule, style_rule=style_rule,
        intents="\n".join(f"- {i}: {config.INTENTS[i]}" for i in prof["intents"]))


def user_prompt(p: Payload, topic: Topic, angle: str, count: int, examples: list[dict],
                do_not_repeat: list[str]) -> str:
    g = p.grounding
    tp = config.TOPIC_PROFILES[topic.code]
    lines = [
        f"TOPIC: {topic.name} ({tp['focus']})",
        f"ANGLE: {angle}",
        f"WRITE: {count} questions",
        "",
        "PRODUCT CONTEXT:",
        f"- Brand: {g.brand}",
        f"- Condition(s): {', '.join(g.diseaseAreas) or 'not specified'}",
        f"- Stage: {g.stage}; market access: {g.marketAccess or 'unknown'}",
    ]
    if g.websites.corporate:
        lines.append(f"- Company website: {g.websites.corporate} (the company name is not given; do not guess it)")
    if p.options.includeCompetitorPrompts and g.competitors:
        lines.append(f"- Competitors: {', '.join(g.competitors)}")
        if angle.startswith("vs "):
            lines.append(f'- Every question compares {g.brand} with {angle[3:]}; set "competitor" to "{angle[3:]}".')
        else:
            lines.append('- Set "competitor" to the competitor named in the question, else null.')
    else:
        lines.append('- Do not mention competitors; "competitor" is always null.')
    lines.append(f"- Name {g.brand} in most questions." if tp["brand"] == "most" else
                 f"- Most questions are about the condition in general; name {g.brand} in only some of them.")
    if examples:
        lines += ["", "EXAMPLE SEARCHES (style reference only):", *[f"- {e['text']}" for e in examples]]
    lines += ["", "DO NOT REPEAT (or paraphrase):", *([f"- {t}" for t in do_not_repeat] or ["- (none)"])]
    return "\n".join(lines)


def dry_run(p: Payload, seed: int | None = None) -> None:
    """Print the system prompt per audience and the user prompt per pair, without calling the LLM.
    Patient examples come from the fallback list (no harvesting)."""
    alloc = p.allocate(seed)
    orders = angle_orders(p, alloc, seed)
    sampler = Sampler(build_pool([], p), seed)
    for aud in p.active_audiences():
        print(f"\n{'#' * 100}\n# SYSTEM PROMPT: {aud.code}\n{'#' * 100}\n{system_prompt(p, aud)}")
        for topic in p.active_topics():
            q = alloc[aud.code, topic.code]
            print(f"\n----- {aud.code} x {topic.code}: quota {q} -----")
            if not q:
                print("(skipped: no prompts allocated this run)")
                continue
            angle = orders[aud.code, topic.code][0]
            examples = sampler.pick(topic.code) if config.AUDIENCE_PROFILES[aud.code]["fewshot"] else []
            print(user_prompt(p, topic, angle, math.ceil(q * config.OVERGENERATE), examples,
                              p.do_not_repeat(aud.code)))
    print(f"\nAllocation: {sum(alloc.values())} prompts over {len(alloc)} pairs")
    for w in p.warnings():
        print("WARNING:", w)


# ---- run ----


@dataclass
class RunResult:
    output: Output
    records: list[dict]          # accepted prompts with debug fields (angle, batchId, referenceIds)
    allocation: dict[tuple[str, str], int]
    pool: list[dict]
    harvest: HarvestResult
    batches: list[dict] = field(default_factory=list)  # per-call log
    warnings: list[str] = field(default_factory=list)
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


def generate(p: Payload, use_autocomplete: bool | None = None, use_trends: bool | None = None,
             model: str | None = None, seed: int | None = None, on_event=print, offline: bool = False) -> RunResult:
    t0 = time.time()
    alloc = p.allocate(seed)
    orders = angle_orders(p, alloc, seed)
    pairs = [(a, t) for a, t in p.pairs() if alloc[a.code, t.code]]
    on_event(f"Allocation: {sum(alloc.values())} prompts over {len(pairs)} pairs")

    # harvested Google searches are style examples for few-shot audiences only (patient)
    h = HarvestResult()
    pool: list[dict] = []
    if any(config.AUDIENCE_PROFILES[a.code]["fewshot"] for a in p.active_audiences()):
        h = harvest(p, use_autocomplete, use_trends, offline=offline)
        on_event(f"Harvest: {len(h.questions)} questions, {h.stats}, store {h.stored} rows")
        for e in h.errors:
            on_event(f"Harvest error: {e}")
        pool = build_pool(h.questions, p)
        for i, q in enumerate(pool):
            q["id"] = f"r{i:03d}"
        on_event(f"Reference pool: {len(pool)} ({sum(q['source'] == 'fallback' for q in pool)} fallback)")

    sampler = Sampler(pool, seed)
    dedup = {a.code: Deduper(p.do_not_repeat(a.code)) for a in p.active_audiences()}
    sys_prompts = {a.code: system_prompt(p, a) for a in p.active_audiences()}
    # ponytail: one lock for sampler, dedup and res; pairs run in parallel, batches within a pair in order
    lock = threading.Lock()
    res = RunResult(output=None, records=[], allocation=alloc, pool=pool, harvest=h, warnings=p.warnings())

    def call(aud: Audience, topic: Topic, angle: str, quota: int, batch_id: str) -> list[dict]:
        with lock:
            examples = sampler.pick(topic.code) if config.AUDIENCE_PROFILES[aud.code]["fewshot"] else []
            dnr = list(dedup[aud.code].texts)
        count = math.ceil(quota * config.OVERGENERATE)
        log = {"batchId": batch_id, "audience": aud.code, "topic": topic.code, "angle": angle, "quota": quota,
               "asked": count}
        try:
            r = llm.chat_json(sys_prompts[aud.code], user_prompt(p, topic, angle, count, examples, dnr), model)
        except llm.LLMError as e:
            log["error"] = str(e)
            with lock:
                res.calls += 1
                res.batches.append(log)
                res.warnings.append(f"{batch_id}: LLM call failed: {e}")
            on_event(f"{batch_id} [{angle}] FAILED: {e}")
            return []
        with lock:
            b = filter_batch(r.data.get("prompts"), p, aud.code, dedup[aud.code], quota)
            res.calls += 1
            res.prompt_tokens += r.prompt_tokens
            res.completion_tokens += r.completion_tokens
            log.update(accepted=len(b.accepted), rejected=dict(b.rejected), duplicates=b.duplicates,
                       surplus=b.surplus)
            res.batches.append(log)
        on_event(f"{batch_id} [{angle}] accepted {len(b.accepted)}/{quota}, "
                 f"rejected {sum(b.rejected.values())} {dict(b.rejected) or ''}, duplicates {b.duplicates}")
        refs = [e["id"] for e in examples]
        return [{**x, "audience": aud.code, "topic": topic.code, "angle": angle, "batchId": batch_id,
                 "referenceIds": refs} for x in b.accepted]

    def run_pair(pair: tuple[Audience, Topic]) -> list[dict]:
        aud, topic = pair
        quota = alloc[aud.code, topic.code]
        angles = orders[aud.code, topic.code]
        sizes = batch_sizes(quota)
        name = f"{aud.code}/{topic.code}"
        out: list[dict] = []
        for i, size in enumerate(sizes):
            out += call(aud, topic, angles[i % len(angles)], size, f"{name}-{i + 1}")
        used = {angles[i % len(angles)] for i in range(len(sizes))}
        unused = [a for a in angles if a not in used] or angles
        for j in range(config.TOPUP_CALLS):
            short = quota - len(out)
            if short <= 0:
                break
            out += call(aud, topic, unused[j % len(unused)], short, f"{name}-topup{j + 1}")
        if len(out) < quota:
            with lock:
                res.warnings.append(f"{name}: returned {len(out)} of {quota} after top-up")
        return out

    with ThreadPoolExecutor(max_workers=max(1, min(config.MAX_WORKERS, len(pairs)))) as ex:
        per_pair = list(ex.map(run_pair, pairs))  # map keeps pair order
    res.records = [r for recs in per_pair for r in recs]
    res.output = assemble(p, res.records)
    res.seconds = time.time() - t0
    return res


if __name__ == "__main__":
    import json
    import os
    from collections import Counter

    from promptgen.schema import load_cli

    def extra(ap):
        ap.add_argument("--n", type=int, help="override options.maxPrompts")
        ap.add_argument("--per-pair", type=int, help="override options.promptsPerPair")
        ap.add_argument("--no-sources", action="store_true", help="skip Autocomplete/Trends")
        ap.add_argument("--offline", action="store_true", help="use only stored Autocomplete/Trends results")
        ap.add_argument("--stage", help="override grounding.stage, e.g. Pre-launch")
        ap.add_argument("--seed", type=int)
        ap.add_argument("--model")
        ap.add_argument("--out", help="output path (default output/<requestId>.json)")
        ap.add_argument("--dry-run", action="store_true", help="print the LLM prompts per pair, no calls")

    p, a = load_cli("Generate prompts end to end", extra)
    if a.stage:
        p = p.model_copy(update={"grounding": p.grounding.model_copy(update={"stage": a.stage})})
    opts = {k: v for k, v in {"maxPrompts": a.n, "promptsPerPair": a.per_pair}.items() if v is not None}
    if opts:  # re-validate so CLI overrides get the same limits as the API
        p = Payload.model_validate({**p.model_dump(), "options": {**p.options.model_dump(), **opts}})
    if a.dry_run:
        dry_run(p, a.seed)
        raise SystemExit
    off = False if a.no_sources else None
    r = generate(p, off, off, a.model, a.seed, offline=a.offline)

    print("\n" + "-" * 120)
    for x in r.records:
        print(f"{x['audience'][:9]:9} {x['topic'][:11]:11} {x['intent'][:13]:13} {str(x['competitor'] or ''):13} "
              f"{x['text']}")
    got = Counter((x["audience"], x["topic"]) for x in r.records)
    print("-" * 120)
    topics = [t.code for t in p.active_topics()]
    print("returned/requested per pair:")
    print(f"{'':16}" + "".join(f"{t[:11]:>12}" for t in topics))
    for aud in p.active_audiences():
        print(f"{aud.code:16}" + "".join(f"{got[aud.code, t]}/{r.allocation[aud.code, t]}".rjust(12) for t in topics))
    print(f"requested {sum(r.allocation.values())}, returned {len(r.records)}")
    print(f"{r.calls} LLM calls, tokens in/out {r.prompt_tokens}/{r.completion_tokens}, {r.seconds:.1f}s")
    for w in r.warnings:
        print("WARNING:", w)
    path = a.out or f"output/{p.requestId}.json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(r.output.model_dump(), f, indent=2)
    print(f"wrote {path}")
