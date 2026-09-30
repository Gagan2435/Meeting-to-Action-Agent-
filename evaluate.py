"""Precision / recall / F1 of action-item extraction vs hand-labelled ground truth (LLM vs rule-based baseline)."""
import datetime as dt
import json
import pathlib
import sys

import extract
import llm
from rag import jaccard

HERE = pathlib.Path(__file__).resolve().parent


def score(use_llm):
    labels = json.loads((HERE / "data/labels.json").read_text(encoding="utf-8"))
    tp = fp = fn = 0
    for name, gold in labels.items():
        text = (HERE / "data/transcripts" / name).read_text(encoding="utf-8")
        pred = extract.extract(text, use_llm=use_llm, log=lambda *_: None).action_items
        left = list(gold)
        for p in pred:
            hit = next((g for g in left if g["owner"].lower() == p.owner.lower() and jaccard(g["task"], p.task) >= 0.4), None)
            if hit:
                tp += 1
                left.remove(hit)
            else:
                fp += 1
        fn += len(left)
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return P, R, 2 * P * R / max(P + R, 1e-9), tp, fp, fn


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"{'system':28}{'precision':>10}{'recall':>8}{'F1':>7}   TP/FP/FN")
    runs = [("rule-based baseline", False)] + ([(f"LLM ({llm.provider()})", True)] if llm.provider() != "mock" else [])
    for name, flag in runs:
        P, R, F, tp, fp, fn = score(flag)
        print(f"{name:28}{P:10.2f}{R:8.2f}{F:7.2f}   {tp}/{fp}/{fn}")
    if llm.provider() == "mock":
        print("\nAdd a free GEMINI_API_KEY or GROQ_API_KEY to .env to also score the LLM extractor.")
