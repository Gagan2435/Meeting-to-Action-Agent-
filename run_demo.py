"""One-command pipeline:  transcript -> extract -> RAG dedupe -> plan (MCP tools) -> approve -> execute."""
import argparse
import asyncio
import datetime as dt
import pathlib
import shutil
import sys

import extract
import llm
from rag import Memory

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "output"


def transcribe(path):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("Audio needs:  pip install faster-whisper   (or pass a .txt transcript)")
    segs, _ = WhisperModel("base", compute_type="int8").transcribe(path)
    return " ".join(s.text.strip() for s in segs)


async def run(a):
    import agent  # imported here so `evaluate.py` works even without the mcp package
    if not a.keep and OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(exist_ok=True)
    text = transcribe(a.transcript) if a.transcript.lower().endswith((".mp3", ".wav", ".m4a")) \
        else pathlib.Path(a.transcript).read_text(encoding="utf-8")
    print(f"LLM provider: {llm.provider()}" + ("  (offline rule-based mode; add a free key in .env for the real LLM)" if llm.provider() == "mock" else ""))

    mem = Memory(OUT / "memory.json")
    ctx = mem.context(text)
    print("\n[1/4] RAG: related open tasks from earlier meetings\n" + (ctx or "  (none yet)"))

    m = extract.extract(text, context=ctx)
    dupes = [i for i in m.action_items if mem.is_duplicate(i.task, i.owner)]
    m.action_items = [i for i in m.action_items if i not in dupes]
    print(f"\n[2/4] Extracted from '{m.title}' ({m.date}): {len(m.decisions)} decisions, {len(m.action_items)} new tasks")
    for d in m.decisions:
        print("  decision:", d)
    for i in m.action_items:
        print(f"  task: {i.owner:8} | {i.task} | due {i.due or '-'} | {i.priority}")
    for i in dupes:
        print(f"  skipped duplicate (already tracked): {i.owner} - {i.task}")
    if m.follow_up_date:
        print(f"  follow-up meeting: {m.follow_up_date} {m.follow_up_time}")

    async with agent.MCPHub() as hub:
        print(f"\n[3/4] Agent discovered {len(hub.specs)} MCP tools: " + ", ".join(s["name"] for s in hub.specs))
        calls = agent.make_plan(m, hub.specs)
        for n, c in enumerate(calls, 1):
            print(f"  {n:2}. {c['tool']:13} {c['args'].get('title') or c['args'].get('subject') or ''} {c['args'].get('to', '')}")
        if not calls:
            print("  nothing to do.")
            return
        if not a.yes and input("\nExecute these actions? [y/N] ").strip().lower() != "y":
            print("Cancelled - nothing was executed.")
            return
        print("\n[4/4] Executing via MCP")
        for c in calls:
            try:
                print("  ->", await hub.call(c["tool"], c["args"]))
            except Exception as e:  # noqa
                print("  ! failed:", c["tool"], e)
    for d in m.decisions:
        mem.add("decision", d, m.title, m.date)
    for i in m.action_items:
        mem.add("task", i.task, m.title, m.date, i.owner, i.due)
    mem.save()
    print(f"\nDone. Open the '{OUT.name}/' folder: board.json, calendar/*.ics, outbox/*.txt")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", nargs="?", default=str(HERE / "data/transcripts/product_sync.txt"))
    ap.add_argument("--yes", action="store_true", help="skip the approval prompt")
    ap.add_argument("--keep", action="store_true", help="keep previous output/memory (shows RAG dedupe)")
    asyncio.run(run(ap.parse_args()))
