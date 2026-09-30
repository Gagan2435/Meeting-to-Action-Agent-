"""MCP client + LLM planner. Tools are DISCOVERED from the MCP servers at runtime; the LLM decides which to call."""
import json
import os
import pathlib
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import llm

HERE = pathlib.Path(__file__).resolve().parent
SERVICES = ["calendar", "tasks", "email"]


class MCPHub:
    async def __aenter__(self):
        self.stack, self.sessions, self.specs = AsyncExitStack(), {}, []
        for svc in SERVICES:
            params = StdioServerParameters(command=sys.executable, args=[str(HERE / "servers.py"), svc], env=dict(os.environ))
            r, w = await self.stack.enter_async_context(stdio_client(params))
            s = await self.stack.enter_async_context(ClientSession(r, w))
            await s.initialize()
            for t in (await s.list_tools()).tools:
                self.sessions[t.name] = s
                self.specs.append({"name": t.name, "description": t.description, "schema": t.inputSchema})
        return self

    async def __aexit__(self, *exc):
        await self.stack.aclose()

    async def call(self, name, args):
        res = await self.sessions[name].call_tool(name, args)
        text = "".join(c.text for c in res.content if hasattr(c, "text"))
        return ("ERROR: " + text) if res.isError else text


PLAN_SYSTEM = """You are an operations agent. Given a structured meeting summary and the MCP tools available,
decide which tool calls to make. Return ONLY JSON: {"calls": [{"tool": str, "args": {...}}]}

Rules:
- Use ONLY the tools and argument names in the provided tool list.
- Exactly one create_card per action item (copy owner, due, priority; write a one-line description).
- One create_event for the follow-up meeting if follow_up_date is set (use follow_up_time, attendees = all attendees).
- One send_email per person who owns tasks: friendly, concise, lists ONLY their tasks with due dates, plus the key decisions.
  Address is "<firstname lowercase>@example.com". Never email people with no tasks. Never invent tasks, dates or facts.
- Dates must be ISO YYYY-MM-DD. Do not include any keys other than tool and args."""


def default_plan(m):
    calls = []
    for it in m.action_items:
        calls.append({"tool": "create_card", "args": {
            "title": it.task, "owner": it.owner, "due": it.due or "", "priority": it.priority,
            "description": f"From '{m.title}' ({m.date}). Deadline said: {it.due_text or 'not specified'}."}})
    if m.follow_up_date:
        calls.append({"tool": "create_event", "args": {
            "title": f"Follow-up: {m.title}", "date": m.follow_up_date, "time": m.follow_up_time, "duration_min": 30,
            "attendees": ", ".join(m.attendees), "description": f"Follow-up agreed in '{m.title}'."}})
    for o in sorted({i.owner for i in m.action_items if i.owner != "Unassigned"}):
        mine = [i for i in m.action_items if i.owner == o]
        body = f"Hi {o},\n\nThanks for joining '{m.title}' on {m.date}.\n\n"
        if m.decisions:
            body += "Decisions:\n" + "\n".join(f"- {d}" for d in m.decisions) + "\n\n"
        body += "Your action items:\n" + "\n".join(f"- {i.task} (due {i.due or 'TBD'}, {i.priority} priority)" for i in mine)
        calls.append({"tool": "send_email", "args": {
            "to": f"{o.split()[0].lower()}@example.com", "subject": f"Action items from {m.title}", "body": body + "\n\nBest,\nMeeting Agent"}})
    return calls


def validate(calls, specs):
    """Guardrail: keep only calls to real tools with all required args; strip unknown/None args."""
    by, ok = {s["name"]: s for s in specs}, []
    for c in calls if isinstance(calls, list) else []:
        if not isinstance(c, dict) or c.get("tool") not in by or not isinstance(c.get("args"), dict):
            continue
        sch = by[c["tool"]]["schema"]
        args = {k: v for k, v in c["args"].items() if k in sch.get("properties", {}) and v is not None}
        if all(r in args for r in sch.get("required", [])):
            ok.append({"tool": c["tool"], "args": args})
    return ok


def make_plan(meeting, specs, log=print):
    fallback = default_plan(meeting)
    if llm.provider() == "mock":
        return fallback
    try:
        d = llm.complete_json(PLAN_SYSTEM, json.dumps({"meeting": meeting.model_dump(), "tools": specs}))
        calls = validate(d.get("calls"), specs)
        if sum(c["tool"] == "create_card" for c in calls) < len(meeting.action_items):
            raise ValueError("LLM plan is missing task cards")
        return calls
    except Exception as e:  # noqa
        log(f"  ! LLM planning failed ({e}); using deterministic plan")
        return fallback
