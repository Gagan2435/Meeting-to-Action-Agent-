"""MCP servers. One file, three servers:  python servers.py calendar|tasks|email  (stdio transport)."""
import datetime as dt
import json
import os
import pathlib
import re
import sys
import urllib.parse
import urllib.request

from mcp.server.fastmcp import FastMCP

OUT = pathlib.Path(__file__).resolve().parent / "output"
svc = sys.argv[1] if len(sys.argv) > 1 else ""
mcp = FastMCP(f"meeting-{svc}")


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "item"


if svc == "calendar":
    @mcp.tool()
    def create_event(title: str, date: str, time: str = "10:00", duration_min: int = 30,
                     attendees: str = "", description: str = "") -> str:
        """Create a calendar event saved as an .ics file. date=YYYY-MM-DD, time=HH:MM (24h), attendees=comma separated names."""
        try:
            start = dt.datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        except ValueError:
            return f"ERROR: bad date/time '{date} {time}'"
        end = start + dt.timedelta(minutes=int(duration_min))
        f = "%Y%m%dT%H%M%S"
        lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//meeting-agent//EN", "BEGIN:VEVENT",
                 f"UID:{slug(title)}-{start.strftime(f)}@meeting-agent", f"DTSTAMP:{dt.datetime.now().strftime(f)}",
                 f"DTSTART:{start.strftime(f)}", f"DTEND:{end.strftime(f)}", f"SUMMARY:{title}",
                 f"DESCRIPTION:{description} Attendees: {attendees}".replace("\n", " "), "END:VEVENT", "END:VCALENDAR"]
        d = OUT / "calendar"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{date}_{slug(title)}.ics"
        p.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
        return f"Event '{title}' on {date} {time} saved to {p.relative_to(OUT.parent)}"

elif svc == "tasks":
    def _trello(title, desc, due):
        k, t, l = (os.getenv(x) for x in ("TRELLO_KEY", "TRELLO_TOKEN", "TRELLO_LIST_ID"))
        if not (k and t and l):
            return ""
        try:
            q = urllib.parse.urlencode({"key": k, "token": t, "idList": l, "name": title, "desc": desc, "due": due})
            urllib.request.urlopen(urllib.request.Request("https://api.trello.com/1/cards?" + q, method="POST"), timeout=15)
            return " (also created in Trello)"
        except Exception as e:  # noqa
            return f" (Trello sync failed: {e})"

    BOARD = OUT / "board.json"

    def _load():
        return json.loads(BOARD.read_text(encoding="utf-8")) if BOARD.exists() else []

    @mcp.tool()
    def create_card(title: str, owner: str = "Unassigned", due: str = "", priority: str = "medium",
                    description: str = "") -> str:
        """Create a task card on the local kanban board (and on Trello if configured). due=YYYY-MM-DD or empty."""
        cards = _load()
        cid = f"TASK-{len(cards) + 1:03d}"
        cards.append({"id": cid, "title": title, "owner": owner, "due": due, "priority": priority,
                      "status": "todo", "description": description})
        OUT.mkdir(parents=True, exist_ok=True)
        BOARD.write_text(json.dumps(cards, indent=2), encoding="utf-8")
        return f"{cid} created for {owner}: {title} (due {due or 'n/a'}){_trello(title, description, due)}"

    @mcp.tool()
    def list_cards() -> str:
        """List all cards currently on the board as JSON."""
        return json.dumps(_load())

elif svc == "email":
    @mcp.tool()
    def send_email(to: str, subject: str, body: str) -> str:
        """Send a follow-up email. In this offline build it is written to output/outbox/ as a .txt file."""
        d = OUT / "outbox"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{len(list(d.glob('*.txt'))) + 1:02d}_{slug(to)}.txt"
        p.write_text(f"To: {to}\nSubject: {subject}\n\n{body}\n", encoding="utf-8")
        return f"Email to {to} saved to {p.relative_to(OUT.parent)}"
else:
    sys.exit("usage: python servers.py calendar|tasks|email")

if __name__ == "__main__":
    mcp.run()
