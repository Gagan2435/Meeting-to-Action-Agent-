"""Tiny RAG layer: BM25 retrieval over past meetings' tasks/decisions (pure Python, no downloads)."""
import json, math, pathlib, re

STOP = set("a an the to of and or for in on at by with is are be will i we you it this that from as our my".split())


def tokens(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


def jaccard(a, b):
    a, b = set(tokens(a)), set(tokens(b))
    return len(a & b) / len(a | b) if a | b else 0.0


class Memory:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.docs = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []

    def add(self, kind, text, meeting, date, owner="", due=None):
        self.docs.append({"kind": kind, "text": text, "meeting": meeting, "date": date, "owner": owner, "due": due})

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.docs, indent=2), encoding="utf-8")

    def search(self, query, k=5):
        q, n = set(tokens(query)), len(self.docs)
        if not n or not q:
            return []
        toks = [tokens(d["text"]) for d in self.docs]
        avg = sum(map(len, toks)) / n or 1
        df = {}
        for t in toks:
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        scored = []
        for d, t in zip(self.docs, toks):
            s = 0.0
            for w in q:
                f = t.count(w)
                if f:
                    idf = math.log(1 + (n - df[w] + .5) / (df[w] + .5))
                    s += idf * f * 2.5 / (f + 1.5 * (.25 + .75 * len(t) / avg))
            if s > 0:
                scored.append((s, d))
        return [d for _, d in sorted(scored, key=lambda x: -x[0])[:k]]

    def context(self, query, k=5):
        hits = [d for d in self.search(query, k * 2) if d["kind"] == "task"][:k]
        return "\n".join(f"- [{d['date']}] {d['owner']}: {d['text']} (due {d['due'] or 'n/a'})" for d in hits)

    def is_duplicate(self, task, owner, thr=0.5):
        return any(d["kind"] == "task" and d["owner"].lower() == owner.lower() and jaccard(d["text"], task) >= thr
                   for d in self.docs)
