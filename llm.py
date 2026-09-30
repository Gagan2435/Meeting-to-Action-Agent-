"""Provider-agnostic LLM client (Gemini / Groq / Ollama) using only the standard library.
Robust to overloaded models: retries with backoff and rotates across several Gemini models."""
import json, os, pathlib, re, time, urllib.error, urllib.request


def load_env():
    p = pathlib.Path(__file__).resolve().parent / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


load_env()

_CACHE = {"models": None}
_BAD = ("image", "tts", "audio", "live", "embed", "thinking", "exp", "vision", "robotics",
        "computer", "learnlm", "gemma", "aqa", "imagen", "veo", "native", "customtools")


def rank_gemini_models(models):
    """Usable text models, best first: stable flash > preview flash > lite > others; newer versions first."""
    names = [m["name"].split("/")[-1] for m in models if "generateContent" in m.get("supportedGenerationMethods", [])]
    ok = [n for n in names if n.startswith("gemini") and not any(b in n for b in _BAD)]

    def rank(n):
        v = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
        return ("flash" in n, "lite" not in n, "preview" not in n, float(v.group(1)) if v else 0, -len(n))
    return sorted(ok, key=rank, reverse=True)


def gemini_models():
    """GEMINI_MODEL from .env if set; otherwise ask the API which models this key can use."""
    if os.getenv("GEMINI_MODEL"):
        return [os.environ["GEMINI_MODEL"]]
    if not _CACHE["models"]:
        req = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                                     headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        with urllib.request.urlopen(req, timeout=30) as r:
            _CACHE["models"] = rank_gemini_models(json.load(r).get("models", []))[:5]
        if not _CACHE["models"]:
            raise RuntimeError("no Gemini model available for this API key")
    return _CACHE["models"]


def provider() -> str:
    p = os.getenv("LLM_PROVIDER", "").lower()
    if p:
        return p
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    return "mock"


def _post(url, body, headers=None):
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}") from None


def parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < 0:
        raise ValueError("no JSON object in LLM output")
    return json.loads(text[a:b + 1])


def _call(p, system, user, attempt):
    if p == "gemini":
        ms = gemini_models()
        d = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{ms[attempt % len(ms)]}:generateContent",
                  {"systemInstruction": {"parts": [{"text": system}]},
                   "contents": [{"role": "user", "parts": [{"text": user}]}],
                   "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}},
                  {"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        return d["candidates"][0]["content"]["parts"][0]["text"]
    if p == "groq":
        d = _post("https://api.groq.com/openai/v1/chat/completions",
                  {"model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), "temperature": 0,
                   "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                  {"Authorization": "Bearer " + os.environ["GROQ_API_KEY"]})
        return d["choices"][0]["message"]["content"]
    if p == "ollama":
        d = _post("http://localhost:11434/api/chat",
                  {"model": os.getenv("OLLAMA_MODEL", "llama3.2"), "stream": False, "format": "json",
                   "options": {"temperature": 0},
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        return d["message"]["content"]
    raise RuntimeError(f"unknown provider {p}")


def complete_json(system: str, user: str) -> dict:
    """Call the configured LLM, return parsed JSON. Retries (503/429/bad JSON) with backoff and model rotation."""
    p = provider()
    if p == "mock":
        raise RuntimeError("mock provider")
    last, tries = None, 6
    for i in range(tries):
        try:
            return parse_json(_call(p, system, user, i))
        except Exception as e:  # noqa
            last = e
            if str(e).startswith(("HTTP 400", "HTTP 401", "HTTP 403")):
                break  # bad key / bad request: retrying will not help
            if i < tries - 1:
                time.sleep(min(2 * (i + 1), 8))
    raise RuntimeError(f"LLM call failed: {last}")