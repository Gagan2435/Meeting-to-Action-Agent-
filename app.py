"""Streamlit UI:  streamlit run app.py"""
import asyncio
import pathlib
import shutil

import pandas as pd
import streamlit as st

import agent
import evaluate
import extract
import llm
from rag import Memory

HERE = pathlib.Path(__file__).resolve().parent
OUT, SAMPLES = HERE / "output", HERE / "data/transcripts"

st.set_page_config(page_title="Meeting-to-Action Agent", layout="wide")
st.title("Meeting-to-Action Agent")
st.caption("LLM extraction + RAG memory + MCP tools + human approval")


def s(v):
    return "" if pd.isna(v) else str(v).strip()


async def _plan(m):
    async with agent.MCPHub() as hub:
        return agent.make_plan(m, hub.specs, log=st.warning)


async def _exec(calls):
    out = []
    async with agent.MCPHub() as hub:
        for c in calls:
            try:
                out.append(await hub.call(c["tool"], c["args"]))
            except Exception as e:  # noqa
                out.append(f"ERROR: {c['tool']}: {e}")
    return out


with st.sidebar:
    st.subheader("Status")
    st.write(f"LLM provider: **{llm.provider()}**")
    if llm.provider() == "mock":
        st.info("Offline rule-based mode. Add a free Gemini/Groq key to `.env` to use a real LLM.")
    if st.button("Reset output and memory"):
        shutil.rmtree(OUT, ignore_errors=True)
        st.session_state.clear()
        st.rerun()

t_run, t_out, t_eval = st.tabs(["Run", "Results", "Evaluation"])

with t_run:
    src = st.radio("Transcript source", ["Sample", "Upload / paste"], horizontal=True)
    if src == "Sample":
        name = st.selectbox("Sample", sorted(p.name for p in SAMPLES.glob("*.txt")))
        base = (SAMPLES / name).read_text(encoding="utf-8")
    else:
        up = st.file_uploader("Transcript (.txt, lines like 'Alice: ...')", type=["txt"])
        base = up.read().decode("utf-8", "replace") if up else ""
    text = st.text_area("Transcript (editable)", base, height=220, key=f"ta_{hash(base)}")

    if st.button("1. Analyze", type="primary") and text.strip():
        mem = Memory(OUT / "memory.json")
        ctx = mem.context(text)
        with st.spinner("Extracting decisions and tasks..."):
            m = extract.extract(text, context=ctx, log=st.warning)
        dupes = [i for i in m.action_items if mem.is_duplicate(i.task, i.owner)]
        m.action_items = [i for i in m.action_items if i not in dupes]
        st.session_state.update(meeting=m, ctx=ctx, dupes=dupes, plan=None, results=None,
                                items0=[i.model_dump() for i in m.action_items])

    if "meeting" in st.session_state:
        m = st.session_state.meeting
        with st.expander("RAG: related open tasks from earlier meetings", expanded=bool(st.session_state.ctx)):
            st.text(st.session_state.ctx or "(none yet)")
        for d in st.session_state.dupes:
            st.warning(f"Skipped duplicate (already tracked): {d.owner} - {d.task}")
        st.subheader(m.title)
        st.write("**Decisions**")
        for d in m.decisions or ["(none)"]:
            st.write("- " + d)
        if m.follow_up_date:
            st.write(f"**Follow-up meeting:** {m.follow_up_date} {m.follow_up_time}")
        st.write("**Action items** (editable: fix, add or delete rows before planning)")
        df = pd.DataFrame(st.session_state.items0, columns=["task", "owner", "due", "priority"])
        edited = st.data_editor(df, num_rows="dynamic", key="items")
        items = [extract.ActionItem(task=s(r.task), owner=s(r.owner) or "Unassigned", due=s(r.due) or None,
                                    priority=s(r.priority) if s(r.priority) in ("low", "medium", "high") else "medium")
                 for r in edited.itertuples() if s(r.task)]
        m2 = m.model_copy(update={"action_items": items})

        if st.button("2. Plan actions via MCP"):
            with st.spinner("Starting MCP servers and planning..."):
                st.session_state.plan = asyncio.run(_plan(m2))
            st.session_state.results = None
        plan = st.session_state.get("plan")
        if plan:
            st.caption("If you edit the tasks above, click 'Plan actions' again.")
            pdf = pd.DataFrame({"run": [True] * len(plan), "tool": [c["tool"] for c in plan],
                                "summary": [c["args"].get("title") or c["args"].get("subject") or "" for c in plan],
                                "to / owner": [c["args"].get("to") or c["args"].get("owner") or "" for c in plan]})
            sel = st.data_editor(pdf, disabled=["tool", "summary", "to / owner"], key="plan_sel")
            with st.expander("Raw MCP tool calls"):
                st.json(plan)
            if st.button("3. Approve and execute", type="primary"):
                chosen = [c for c, r in zip(plan, sel["run"]) if r]
                with st.spinner("Executing via MCP..."):
                    st.session_state.results = asyncio.run(_exec(chosen))
                mem = Memory(OUT / "memory.json")
                for d in m2.decisions:
                    mem.add("decision", d, m2.title, m2.date)
                for i in m2.action_items:
                    mem.add("task", i.task, m2.title, m2.date, i.owner, i.due)
                mem.save()
        if st.session_state.get("results"):
            st.success("Done. See the Results tab.")
            st.code("\n".join(st.session_state.results))

with t_eval:
    st.write("Precision / recall / F1 of action-item extraction on `data/labels.json`.")
    if st.button("Run evaluation"):
        rows = [("rule-based baseline", False)] + ([(f"LLM ({llm.provider()})", True)] if llm.provider() != "mock" else [])
        res = []
        with st.spinner("Scoring..."):
            for n, f in rows:
                P, R, F, tp, fp, fn = evaluate.score(f)
                res.append({"system": n, "precision": round(P, 2), "recall": round(R, 2), "F1": round(F, 2), "TP": tp, "FP": fp, "FN": fn})
        st.dataframe(pd.DataFrame(res))

with t_out:  # rendered last so it shows files created in this run
    import json
    board = OUT / "board.json"
    st.subheader("Task board")
    if board.exists():
        st.dataframe(pd.DataFrame(json.loads(board.read_text(encoding="utf-8"))))
    else:
        st.write("No tasks yet.")
    st.subheader("Calendar invites")
    for p in sorted((OUT / "calendar").glob("*.ics")):
        st.download_button(f"Download {p.name}", p.read_bytes(), file_name=p.name, key=f"ics_{p.name}")
    st.subheader("Outbox")
    for p in sorted((OUT / "outbox").glob("*.txt")):
        with st.expander(p.name):
            st.text(p.read_text(encoding="utf-8"))
