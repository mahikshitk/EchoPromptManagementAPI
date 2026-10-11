"""Allocation, the per-topic batch loop, top-up, and the end-to-end run."""

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from promptgen import config, llm
from promptgen.fewshot import Sampler, build_pool
from promptgen.harvest import HarvestResult, harvest
from promptgen.postprocess import Deduper, assemble, filter_batch
from promptgen.schema import Output, Payload, Topic

# ---- allocation ----


def allocate(n: int, codes: list[str], floor: int) -> dict[str, int]:
    """Largest-remainder split of n by config weights, each topic at least `floor` (if n allows)."""
    w = {c: config.TOPIC_WEIGHTS[c] for c in codes}
    total = sum(w.values())
    exact = {c: n * w[c] / total for c in codes}
    q = {c: int(exact[c]) for c in codes}
    for c in sorted(codes, key=lambda c: exact[c] - q[c], reverse=True)[: n - sum(q.values())]:
        q[c] += 1
    floor = min(floor, n // len(codes))
    for c in codes:
        while q[c] < floor:  # take from the biggest topic that stays above floor
            donor = max(codes, key=lambda d: q[d])
            q[donor] -= 1
            q[c] += 1
    return q


def batch_sizes(q: int) -> list[int]:
    """ceil(q/BATCH_SIZE) near-equal batches, e.g. 19 -> [10, 9]."""
    if q <= 0:
        return []
    k = math.ceil(q / config.BATCH_SIZE)
    return [q // k + (1 if i < q % k else 0) for i in range(k)]


def angles_for(topic: str, p: Payload) -> list[str]:
    """Topic angles with "vs competitor" expanded per competitor, or dropped if comparisons are off."""
    out = []
    for a in config.ANGLES[topic]:
        if a != config.VS_COMPETITOR:
            out.append(a)
        elif p.options.includeCompetitorPrompts:
            out += [f"vs {c}" for c in p.grounding.competitors]
    return out


# ---- prompts ----

SYSTEM = """You write the questions a real patient would type into ChatGPT, Gemini or Perplexity about a medicine.

Voice: {voice}.
Market: {market}. Language: {language}. Use framing natural for that market (for the UK: GP, NHS, prescription charge).

Rules:
- Write questions only, never answers. Every question ends with "?".
- First person, lay language, no medical jargon.
- Match the phrasing style and length of the EXAMPLE SEARCHES, but never copy them.
- Do not invent symptoms, doses, numbers, trial results or personal histories.
- Comparisons are questions, never claims.
{stage_rule}- Never repeat or paraphrase anything in the DO NOT REPEAT list. Every question must be distinct from the others.
- Every question is about {brand}; name it in most questions.

Reply with JSON only:
{{"prompts": [{{"text": "...", "intent": "<one of: {intents}>", "competitor": "<exact competitor name or null>"}}]}}"""


def system_prompt(p: Payload) -> str:
    g = p.grounding
    stage_rule = "" if g.stage.lower() == "launched" else (
        f"- {g.brand} is not launched yet (stage: {g.stage}). Never imply it is approved, available, "
        "prescribed, or that anyone is already taking it.\n")
    return SYSTEM.format(voice=p.audience.voice or "lay, first person", market=g.market.name or g.market.isoCode,
                         language=p.options.language, stage_rule=stage_rule, brand=g.brand,
                         intents=", ".join(config.INTENTS))


def user_prompt(p: Payload, topic: Topic, angle: str, count: int, examples: list[dict],
                do_not_repeat: list[str]) -> str:
    g = p.grounding
    lines = [
        f"TOPIC: {topic.name}",
        f"ANGLE: {angle}",
        f"WRITE: {count} questions",
        "",
        "PRODUCT CONTEXT:",
        f"- Brand: {g.brand}",
        f"- Condition(s): {', '.join(g.diseaseAreas) or 'not specified'}",
        f"- Stage: {g.stage}; market access: {g.marketAccess or 'unknown'}",
    ]
    if p.options.includeCompetitorPrompts and g.competitors:
        lines.append(f"- Competitors: {', '.join(g.competitors)}")
        if angle.startswith("vs "):
            lines.append(f'- Every question compares {g.brand} with {angle[3:]}; set "competitor" to "{angle[3:]}".')
        else:
            lines.append('- Set "competitor" to the competitor named in the question, else null.')
    else:
        lines.append('- Do not mention competitors; "competitor" is always null.')
    lines += ["", "EXAMPLE SEARCHES (style reference only):", *[f"- {e['text']}" for e in examples]]
    lines += ["", "DO NOT REPEAT (or paraphrase):", *([f"- {t}" for t in do_not_repeat] or ["- (none)"])]
    return "\n".join(lines)


# ---- run ----


@dataclass
class RunResult:
    output: Output
    records: list[dict]          # accepted prompts with debug fields (angle, batchId, referenceIds)
    allocation: dict[str, int]
    pool: list[dict]
    harvest: HarvestResult
    batches: list[dict] = field(default_factory=list)  # per-call log
    warnings: list[str] = field(default_factory=list)
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


def generate(p: Payload, n: int | None = None, use_autocomplete: bool | None = None,
             use_trends: bool | None = None, model: str | None = None, seed: int | None = None,
             on_event=print) -> RunResult:
    t0 = time.time()
    n = n or p.options.maxPrompts
    if not 1 <= n <= 100:
        raise ValueError("n must be between 1 and 100")
    topics = p.enabled_topics()
    alloc = allocate(n, [t.code for t in topics], p.options.promptsPerPair)
    on_event(f"Allocation for n={n}: {alloc}")

    h = harvest(p, use_autocomplete, use_trends)
    on_event(f"Harvest: {len(h.questions)} questions, {h.stats}")
    pool = build_pool(h.questions, p)
    for i, q in enumerate(pool):
        q["id"] = f"r{i:03d}"
    on_event(f"Reference pool: {len(pool)} ({sum(q['source'] == 'fallback' for q in pool)} fallback)")

    sampler = Sampler(pool, seed)
    dedup = Deduper(p.do_not_repeat())
    # ponytail: one lock for sampler + dedup; topics run in parallel, batches within a topic in order
    lock = threading.Lock()
    sys_prompt = system_prompt(p)
    res = RunResult(output=None, records=[], allocation=alloc, pool=pool, harvest=h, warnings=p.warnings())

    def call(topic: Topic, angle: str, quota: int, batch_id: str) -> list[dict]:
        with lock:
            examples = sampler.pick(topic.code)
            dnr = list(dedup.texts)
        count = math.ceil(quota * config.OVERGENERATE)
        log = {"batchId": batch_id, "topic": topic.code, "angle": angle, "quota": quota, "asked": count}
        try:
            r = llm.chat_json(sys_prompt, user_prompt(p, topic, angle, count, examples, dnr), model)
        except llm.LLMError as e:
            log["error"] = str(e)
            with lock:
                res.calls += 1
                res.batches.append(log)
                res.warnings.append(f"{batch_id}: LLM call failed: {e}")
            on_event(f"{batch_id} [{angle}] FAILED: {e}")
            return []
        with lock:
            b = filter_batch(r.data.get("prompts"), p, dedup, quota)
            res.calls += 1
            res.prompt_tokens += r.prompt_tokens
            res.completion_tokens += r.completion_tokens
            log.update(accepted=len(b.accepted), rejected=dict(b.rejected), duplicates=b.duplicates,
                       surplus=b.surplus)
            res.batches.append(log)
        on_event(f"{batch_id} [{angle}] accepted {len(b.accepted)}/{quota}, "
                 f"rejected {sum(b.rejected.values())} {dict(b.rejected) or ''}, duplicates {b.duplicates}")
        refs = [e["id"] for e in examples]
        return [{**x, "topic": topic.code, "angle": angle, "batchId": batch_id, "referenceIds": refs}
                for x in b.accepted]

    def run_topic(topic: Topic) -> list[dict]:
        quota = alloc[topic.code]
        angles = angles_for(topic.code, p)
        out: list[dict] = []
        for i, size in enumerate(batch_sizes(quota)):
            out += call(topic, angles[i % len(angles)], size, f"{topic.code}-{i + 1}")
        used = {r["angle"] for r in out} | {angles[i % len(angles)] for i in range(len(batch_sizes(quota)))}
        unused = [a for a in angles if a not in used] or angles
        for j in range(config.TOPUP_CALLS):
            short = quota - len(out)
            if short <= 0:
                break
            out += call(topic, unused[j % len(unused)], short, f"{topic.code}-topup{j + 1}")
        if len(out) < quota:
            with lock:
                res.warnings.append(f"{topic.code}: returned {len(out)} of {quota} after top-up")
        return out

    with ThreadPoolExecutor(max_workers=len(topics)) as ex:
        per_topic = list(ex.map(run_topic, topics))  # map keeps topic order
    res.records = [r for recs in per_topic for r in recs]
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
        ap.add_argument("--no-sources", action="store_true", help="skip Autocomplete/Trends")
        ap.add_argument("--stage", help="override grounding.stage, e.g. Pre-launch")
        ap.add_argument("--seed", type=int)
        ap.add_argument("--model")
        ap.add_argument("--out", help="output path (default output/<requestId>.json)")

    p, a = load_cli("Generate patient prompts end to end", extra)
    if a.stage:
        p = p.model_copy(update={"grounding": p.grounding.model_copy(update={"stage": a.stage})})
    off = False if a.no_sources else None
    r = generate(p, a.n, off, off, a.model, a.seed)

    print("\n" + "-" * 100)
    for x in r.records:
        print(f"{x['topic']:11} {x['intent']:13} {str(x['competitor'] or ''):13} {x['angle'][:22]:22} {x['text']}")
    got = Counter(x["topic"] for x in r.records)
    print("-" * 100)
    print(f"requested {sum(r.allocation.values())} {r.allocation}, returned {len(r.records)} {dict(got)}")
    print(f"{r.calls} LLM calls, tokens in/out {r.prompt_tokens}/{r.completion_tokens}, {r.seconds:.1f}s")
    for w in r.warnings:
        print("WARNING:", w)
    path = a.out or f"output/{p.requestId}.json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(r.output.model_dump(), f, indent=2)
    print(f"wrote {path}")
