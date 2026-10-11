"""Streamlit demo: payload in, patient prompts out."""

import hashlib
import json
import os
import queue
import threading

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from promptgen import llm
from promptgen.generator import allocate, generate
from promptgen.schema import Payload

st.set_page_config(page_title="Patient Prompt Agent", layout="wide")
st.title("Patient Prompt Agent")

if not os.getenv("OPENROUTER_API_KEY"):
    st.error("OPENROUTER_API_KEY is not set. Add it to .env and restart.")

# ---- payload ----
if "payload_text" not in st.session_state:
    st.session_state.payload_text = open("sample_payload.json").read()

left, right = st.columns([3, 1])
with left:
    st.text_area("Payload (JSON)", key="payload_text", height=320)

try:
    payload = Payload.model_validate_json(st.session_state.payload_text)
except ValidationError as e:
    st.error(f"Invalid payload:\n\n```\n{e}\n```")
    st.stop()

g = payload.grounding
# reset override widgets whenever the payload's grounding changes
gkey = hashlib.md5(g.model_dump_json().encode()).hexdigest()[:8]

# ---- sidebar ----
with st.sidebar:
    st.header("Settings")
    model = st.text_input("Model", llm.default_model())
    use_ac = st.toggle("Google Autocomplete", payload.options.includeAutocomplete, key=f"ac{gkey}")
    use_trends = st.toggle("Google Trends", payload.options.includeGoogleTrends, key=f"tr{gkey}")
    debug = st.toggle("Debug", False)

    st.header("Grounding")
    st.caption("Pre-filled from the payload. Edit to generate for another brand.")
    brand = st.text_input("Brand", g.brand, key=f"brand{gkey}")
    diseases = st.text_input("Disease areas (comma-separated)", ", ".join(g.diseaseAreas), key=f"dis{gkey}")
    competitors = st.text_input("Competitors (comma-separated)", ", ".join(g.competitors), key=f"comp{gkey}")
    stage = st.text_input("Stage", g.stage, key=f"stage{gkey}",
                          help='Anything other than "Launched" forbids availability/approval wording.')

with right:
    n = st.number_input("Number of prompts (n)", 1, 100, payload.options.maxPrompts, key=f"n{gkey}")
    st.caption(f"Topic split: {allocate(int(n), [t.code for t in payload.enabled_topics()], payload.options.promptsPerPair)}")
    go = st.button("Generate", type="primary", width="stretch", disabled=not brand.strip())

for w in payload.warnings():
    st.warning(w)

# ---- run ----
if go:
    split = lambda s: [x for x in s.split(",")]
    p = payload.with_grounding(brand, split(diseases), split(competitors))
    p = p.model_copy(update={"grounding": p.grounding.model_copy(update={"stage": stage.strip() or "Launched"})})

    events: queue.Queue = queue.Queue()
    box: dict = {}

    def work():  # generate() reports from worker threads; Streamlit can only draw from this script's thread
        try:
            box["result"] = generate(p, int(n), use_ac, use_trends, model.strip() or None, on_event=events.put)
        except Exception as e:  # surfaced below
            box["error"] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()
    with st.status(f"Generating {n} prompts for {p.grounding.brand}...", expanded=True) as status:
        while t.is_alive() or not events.empty():
            try:
                st.write(events.get(timeout=0.2))
            except queue.Empty:
                pass
        if "error" in box:
            status.update(label=f"Failed: {box['error']}", state="error")
        else:
            r = box["result"]
            status.update(label=f"Done: {len(r.records)}/{n} prompts in {r.seconds:.1f}s", state="complete",
                          expanded=False)
            st.session_state.result = r
            st.session_state.result_brand = p.grounding.brand
    if "error" in box:
        st.exception(box["error"])

# ---- results (kept across reruns, e.g. after download) ----
r = st.session_state.get("result")
if r:
    st.divider()
    st.subheader(f"Results for {st.session_state.result_brand}")
    c = st.columns(5)
    c[0].metric("Requested", sum(r.allocation.values()))
    c[1].metric("Returned", len(r.records))
    c[2].metric("LLM calls", r.calls)
    c[3].metric("Tokens in / out", f"{r.prompt_tokens:,} / {r.completion_tokens:,}")
    c[4].metric("Time", f"{r.seconds:.1f}s")

    got = pd.Series([x["topic"] for x in r.records]).value_counts()
    st.caption(" · ".join(f"{t}: {got.get(t, 0)}/{q}" for t, q in r.allocation.items()))
    for w in r.warnings:
        if w not in payload.warnings():
            st.warning(w)

    out = r.output.model_dump()
    rows = []
    for o, rec in zip(out["prompts"], r.records):
        row = {"tempId": o["tempId"], "text": o["text"], "topic": o["topic"]["code"], "intent": o["intent"],
               "competitor": o["competitor"], "groundingUsed": ", ".join(o["groundingUsed"])}
        if debug:
            row |= {"angle": rec["angle"], "batchId": rec["batchId"], "referenceIds": ", ".join(rec["referenceIds"])}
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.download_button("Download JSON (client schema)", json.dumps(out, indent=2),
                       file_name=f"{out['requestId']}_{out['runId']}.json", mime="application/json",
                       on_click="ignore")

    with st.expander(f"LLM batches ({len(r.batches)})"):
        st.dataframe(pd.DataFrame(r.batches), width="stretch", hide_index=True)
    with st.expander(f"Reference pool ({len(r.pool)}) — harvest {r.harvest.stats}"):
        for e in r.harvest.errors:
            st.warning(e)
        st.dataframe(pd.DataFrame(r.pool)[["id", "text", "topic", "source", "seed"]],
                     width="stretch", hide_index=True)
    with st.expander("Output JSON"):
        st.json(out)
