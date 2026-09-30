# Meeting-to-Action Agent (MCP + LLM + RAG)

Turns a meeting transcript into real actions: task cards, calendar invites (.ics) and follow-up emails.

## Run (2 commands, Python 3.10+, no keys needed)
```
pip install -r requirements.txt
python run_demo.py            # add --yes to skip the approval prompt
python evaluate.py            # precision / recall / F1
streamlit run app.py          # web UI: analyze, edit tasks, approve, view results
```
RAG demo: `python run_demo.py` then `python run_demo.py data/transcripts/progress_review.txt --keep`
(the second meeting retrieves earlier tasks and skips the duplicate one).

Results appear in `output/`: `board.json`, `calendar/*.ics`, `outbox/*.txt`.

## Use a real LLM (free)
Copy `.env.example` to `.env`, paste a free Gemini or Groq key (or run Ollama locally). Nothing else changes.
Without a key it runs a rule-based extractor, which is also the baseline in the eval.
Optional: `TRELLO_*` vars push cards to a real Trello list. Audio: `pip install faster-whisper`, pass a .mp3/.wav.

## Architecture
transcript -> **RAG** (BM25 over past meetings) -> **LLM extraction** (few-shot prompt, JSON, Pydantic validation)
-> date resolution + dedupe -> **agent** discovers tools from 3 **MCP servers** (calendar, tasks, email) and plans calls
-> guardrail validation -> **human approval** -> execution via MCP.

| Topic | Where |
|---|---|
| MCP servers / client | `servers.py` / `agent.py` |
| Prompt engineering | `extract.py` (EXTRACT_SYSTEM), `agent.py` (PLAN_SYSTEM) |
| RAG | `rag.py` |
| Agentic tool use | `agent.py` |
| Evaluation | `evaluate.py`, `data/labels.json` |

## Resume bullets (fill in your own numbers from `python evaluate.py`)
- Built an agentic meeting-to-action system using MCP (3 tool servers, runtime tool discovery) and an LLM planner that turns transcripts into task cards, calendar events and follow-up emails, with human-in-the-loop approval.
- Designed few-shot, schema-constrained prompts with Pydantic validation and a deterministic fallback; improved extraction F1 from X (rule-based) to Y (LLM) on a hand-labelled set.
- Added a BM25 retrieval memory of past meetings to give the LLM context and skip duplicate tasks across meetings.
