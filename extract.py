"""Structured extraction: LLM (few-shot prompt + JSON schema + Pydantic validation) with a rule-based fallback."""
import datetime as dt
import re
from typing import Optional

from pydantic import BaseModel, Field

import llm
from rag import jaccard

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MON = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
MON_RE = "(?:" + "|".join(MON) + ")[a-z]*\\.?"


class ActionItem(BaseModel):
    task: str
    owner: str = "Unassigned"
    due_text: str = ""
    due: Optional[str] = None
    priority: str = "medium"


class Meeting(BaseModel):
    title: str = "Meeting"
    date: str
    attendees: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    follow_up_text: str = ""
    follow_up_date: Optional[str] = None
    follow_up_time: str = "10:00"


# ---------------------------------------------------------------- dates
def resolve_date(text: str, ref: dt.date) -> Optional[str]:
    """Turn 'next Friday', 'by tomorrow', 'October 15', 'in 3 days' into ISO dates relative to `ref`."""
    t, td = text.lower(), dt.timedelta
    m = re.search(r"\d{4}-\d{2}-\d{2}", t)
    if m:
        return m.group(0)
    if re.search(r"\b(today|eod|end of (the )?day)\b", t):
        return ref.isoformat()
    if "tomorrow" in t:
        return (ref + td(days=1)).isoformat()
    m = re.search(r"in (\d+) (day|week)s?", t)
    if m:
        return (ref + td(days=int(m.group(1)) * (7 if m.group(2) == "week" else 1))).isoformat()
    if re.search(r"end of (the )?week|\beow\b", t):
        return (ref + td(days=(4 - ref.weekday()) % 7)).isoformat()
    if "next week" in t:
        return (ref - td(days=ref.weekday()) + td(days=7)).isoformat()
    for i, d in enumerate(DAYS):
        m = re.search(rf"\b(next\s+)?{d}\b", t)
        if m:
            if m.group(1):  # "next X" = X in the following calendar week
                return (ref - td(days=ref.weekday()) + td(days=7 + i)).isoformat()
            return (ref + td(days=(i - ref.weekday()) % 7 or 7)).isoformat()
    m = re.search(rf"\b({MON_RE})\s+(\d{{1,2}})\b", t) or None
    if m:
        mon, day = m.group(1), int(m.group(2))
    else:
        m2 = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MON_RE})", t)
        if not m2:
            return None
        day, mon = int(m2.group(1)), m2.group(2)
    try:
        d = dt.date(ref.year, MON.index(mon[:3]) + 1, day)
        return (d if d >= ref else d.replace(year=ref.year + 1)).isoformat()
    except ValueError:
        return None


def parse_time(text: str) -> str:
    m = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text.lower())
    if not m:
        return "10:00"
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ap == "pm" and h < 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    return f"{h:02d}:{mi:02d}" if h < 24 and mi < 60 else "10:00"


# ---------------------------------------------------------------- transcript
def parse_transcript(text, default_ref):
    title, ref, turns = "Meeting", default_ref, []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"(?i)^title:\s*(.+)$", line)
        if m:
            title = m.group(1)
            continue
        m = re.match(r"(?i)^date:\s*(\d{4}-\d{2}-\d{2})", line)
        if m:
            ref = dt.date.fromisoformat(m.group(1))
            continue
        m = re.match(r"^([A-Za-z][\w .'-]{0,30}):\s*(.+)$", line)
        turns.append((m.group(1).strip(), m.group(2)) if m else ("", line))
    return title, ref, turns


# ---------------------------------------------------------------- rule-based baseline
_DAY = "|".join(DAYS)
DUE = re.compile(rf"\s*\b(?:by|before|until|due|on|for)\s+((?:next\s+)?(?:{_DAY})|tomorrow|today|eod|end of (?:the )?(?:week|day)|eow"
                 rf"|\d{{4}}-\d{{2}}-\d{{2}}|{MON_RE}\s+\d{{1,2}}(?:st|nd|rd|th)?|\d{{1,2}}(?:st|nd|rd|th)?\s+{MON_RE}"
                 rf"|in \d+ (?:days|weeks)|next week)\b", re.I)
COMMIT = re.compile(r"\bI(?:'ll| will| can| am going to|'m going to)\s+(.+)|\blet me\s+(.+)", re.I)
ASK = re.compile(r"^([A-Z][a-z]+),\s+(?:can|could|would|will)\s+you\s+(?:please\s+)?(.+)")
DECISION = re.compile(r"\bwe(?:'ve)? (?:decided|agreed)\b|\bdecision:|\blet's go with\b|\bwe(?:'ll| will) go with\b", re.I)
FOLLOW = re.compile(r"\b(meet again|follow-up meeting|next sync|catch up|reconvene|regroup)\b", re.I)
HIGH = re.compile(r"[,;]?\s*(?:it'?s |this is )?(?:urgent|asap|critical|a blocker|top priority)\b", re.I)
LOW = re.compile(r"[,;]?\s*(?:when (?:i|you) get (?:some )?time|no rush|low priority|nice to have)\b", re.I)


def _clean(task):
    task = re.sub(r"^(?:also|then|just)\s+", "", task.strip(), flags=re.I)
    task = re.sub(r"\s+([,.])", r"\1", task)
    task = re.sub(r"[,;]?\s*(?:too|still on track)$", "", task.strip(" ,.;!?"), flags=re.I).strip(" ,.;!?")
    return task


def mock_extract(turns):
    decisions, items, fu = [], [], ""
    for speaker, utt in turns:
        for s in re.split(r"(?<=[.?!])\s+", utt):
            if FOLLOW.search(s) and resolve_date(s, dt.date(2000, 1, 3)):
                fu = s
            elif DECISION.search(s):
                decisions.append(re.sub(r"(?i)^decision:\s*", "", s).strip())
            else:
                m, owner, body = ASK.match(s), speaker, None
                if m:
                    owner, body = m.group(1), m.group(2)
                else:
                    m = COMMIT.search(s)
                    if m and speaker:
                        body = m.group(1) or m.group(2)
                if not body:
                    continue
                prio = "high" if HIGH.search(body) else "low" if LOW.search(body) else "medium"
                due = DUE.search(body)
                due_text = due.group(1) if due else ""
                body = LOW.sub("", HIGH.sub("", DUE.sub("", body, count=1)))
                items.append(ActionItem(task=_clean(body), owner=owner, due_text=due_text, priority=prio))
    return decisions, items, fu


# ---------------------------------------------------------------- LLM extraction
EXTRACT_SYSTEM = """You are a meticulous meeting analyst. Read the transcript and return ONLY a JSON object:
{"decisions": [str], "action_items": [{"task": str, "owner": str, "due_text": str, "priority": "low|medium|high"}], "follow_up": str}

Rules:
1. An action item is an explicit commitment or assignment ("I'll...", "Sam, can you...", "that's on Marcus"). Never invent tasks.
2. "owner" is the person's first name as written in the transcript; use "Unassigned" if nobody owns it.
3. "due_text" is the deadline phrase verbatim ("by Friday", "before October 15"); "" if none.
4. priority is "high" only if urgency is stated (urgent, blocker, ASAP), "low" if the speaker says it is optional/no rush, else "medium".
5. "decisions" are choices the group made, not tasks. "follow_up" is the sentence proposing the next meeting ("" if none).
6. Do NOT repeat tasks already listed under 'Open tasks from earlier meetings' unless the transcript changes them.
7. Keep each task short, imperative, and self-contained (verb first).

Example
Transcript: "Lee: We decided to use Postgres. Ana, can you write the migration by Monday? It's urgent. Lee: Let's regroup next Thursday at 2pm."
Output: {"decisions": ["We decided to use Postgres."], "action_items": [{"task": "Write the database migration", "owner": "Ana", "due_text": "by Monday", "priority": "high"}], "follow_up": "Let's regroup next Thursday at 2pm."}"""


def extract(text, ref=None, context="", use_llm=True, log=print):
    title, ref, turns = parse_transcript(text, ref or dt.date.today())
    attendees = sorted({s for s, _ in turns if s})
    m = None
    if use_llm and llm.provider() != "mock":
        try:
            d = llm.complete_json(EXTRACT_SYSTEM, f"Meeting date: {ref}\nOpen tasks from earlier meetings:\n"
                                                  f"{context or 'none'}\n\nTRANSCRIPT:\n{text}")
            items = []
            for a in d.get("action_items", []):
                pr = str(a.get("priority") or "medium").lower()
                items.append(ActionItem(task=str(a["task"]).strip(), owner=str(a.get("owner") or "Unassigned"),
                                        due_text=str(a.get("due_text") or ""), priority=pr if pr in ("low", "medium", "high") else "medium"))
            m = Meeting(title=title, date=ref.isoformat(), attendees=attendees, action_items=items,
                        decisions=[str(x) for x in d.get("decisions", [])], follow_up_text=str(d.get("follow_up") or ""))
        except Exception as e:  # noqa
            log(f"  ! LLM extraction failed ({e}); using rule-based fallback")
    if m is None:
        dec, items, fu = mock_extract(turns)
        m = Meeting(title=title, date=ref.isoformat(), attendees=attendees, decisions=dec, action_items=items, follow_up_text=fu)
    return finalize(m, ref)


def finalize(m: Meeting, ref):
    """Deterministic post-processing: resolve dates against the meeting date and drop duplicates."""
    kept = []
    for it in m.action_items:
        it.due = resolve_date(it.due_text, ref) if it.due_text else None
        if not any(k.owner == it.owner and jaccard(k.task, it.task) >= 0.7 for k in kept):
            kept.append(it)
    m.action_items = kept
    if m.follow_up_text:
        m.follow_up_date = resolve_date(m.follow_up_text, ref)
        m.follow_up_time = parse_time(m.follow_up_text)
    return m
