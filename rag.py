"""Club Concierge RAG pipeline: load -> chunk -> embed -> retrieve -> answer with citations."""
import os
from pathlib import Path

import numpy as np

DOCS_DIR = Path(__file__).parent / "docs"
REFUSAL = "That isn't in the club's documents."
MODEL = "gemini-3.5-flash-lite"  # override with GEMINI_MODEL in .env
CHUNK_WORDS = 400
OVERLAP_WORDS = 50
TOP_K = 4

SYSTEM_PROMPT = (
    "You are the member concierge for Maple Ridge Golf & Country Club. "
    "Answer only from the context. Cite sources as [Doc – Section]. "
    f"If the context doesn't contain the answer, reply exactly: '{REFUSAL}'"
)


def parse_doc(markdown):
    """Split a markdown document into its title and (section, text) pairs."""
    lines = markdown.splitlines()
    title = lines[0].lstrip("# ").strip()
    sections, name, body = [], None, []
    for line in lines[1:] + ["## "]:  # sentinel heading flushes the last section
        if line.startswith("## "):
            if name and " ".join(body).strip():
                sections.append((name, " ".join(body).strip()))
            name, body = line[3:].strip(), []
        else:
            body.append(line.strip())
    return title, sections


def split_words(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Split text into windows of `size` words that overlap by `overlap` words."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def load_chunks(docs_dir=DOCS_DIR):
    """Load docs/public (anyone) and docs/members (signed-in members only)."""
    chunks = []
    for visibility in ("public", "members"):
        for path in sorted((Path(docs_dir) / visibility).glob("*.md")):
            title, sections = parse_doc(path.read_text(encoding="utf-8"))
            for section, text in sections:
                for piece in split_words(text):
                    chunks.append({"doc": title, "section": section, "text": piece, "visibility": visibility})
    return chunks


def sentence_transformer_embedder(model_name="all-MiniLM-L6-v2"):
    """Local, free embeddings. Returns a function: list of texts -> normalised vectors."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(texts, normalize_embeddings=True)


class Index:
    """Vector index in a NumPy array; cosine similarity = dot product of normalised vectors."""

    def __init__(self, chunks, embed):
        self.chunks, self.embed = chunks, embed
        self.vectors = np.asarray(embed([f"{c['doc']} – {c['section']}: {c['text']}" for c in chunks]))

    def search(self, question, k=TOP_K, member=True):
        """Top-k chunks; visitors who aren't signed in only search public content."""
        scores = self.vectors @ np.asarray(self.embed([question]))[0]
        allowed = [i for i in np.argsort(-scores) if member or self.chunks[i]["visibility"] == "public"]
        return [self.chunks[i] | {"score": float(scores[i])} for i in allowed[:k]]


def gemini_llm(model=None):
    """Returns a function (system, user) -> answer text, backed by the Gemini API (free tier)."""
    from google import genai
    from google.genai import types

    # Reads GEMINI_API_KEY from the environment. Free-tier models are sometimes overloaded
    # (429/503), so retry with exponential backoff instead of failing the member's question.
    client = genai.Client(http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(
        attempts=6, initial_delay=2, max_delay=60, http_status_codes=[429, 500, 503])))
    model = model or os.getenv("GEMINI_MODEL", MODEL)

    def call(system, user, tools=None):
        # With Python functions in `tools`, the SDK runs the tool loop: it calls the
        # functions the model asks for and sends the results back until the model answers.
        response = client.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(system_instruction=system, temperature=0, tools=tools),
        )
        return response.text or REFUSAL  # empty text (e.g. blocked by a safety filter) -> refuse

    return call


TEE_TIME_API = os.getenv("TEE_TIME_API", "http://localhost:5080")
TOOLS_PROMPT = (
    " You can also look up and book tee times with your tools: for questions about open tee times, "
    "use the tools instead of the context. Before booking, repeat the date, time, players and guests "
    "and only book after the member confirms."
)


def tee_time_tools(member_id=None, http=None):
    """Tools the model may call. They use the Tee-Time Booking API (github.com/ridaali62/tee-time-booking).

    Visitors can only look up tee times; booking needs a signed-in member.
    """
    import httpx

    http = http or httpx.Client(base_url=TEE_TIME_API, timeout=10)

    def find_tee_times(date: str, players: int) -> list[dict] | dict:
        """List tee times on a date (YYYY-MM-DD) that still have room for `players` players."""
        try:
            response = http.get("/slots", params={"date": date})
            response.raise_for_status()
        except httpx.HTTPError:
            return {"error": "The tee-time system is not available right now."}
        return [
            {"teeSlotId": s["id"], "time": s["startTime"][11:16], "openPlaces": s["capacity"] - s["bookedPlayers"],
             "weatherDelay": s["weatherDelay"]}
            for s in response.json() if s["capacity"] - s["bookedPlayers"] >= players
        ]

    def book_tee_time(tee_slot_id: int, players: int, guests: int) -> dict:
        """Book a tee time for the signed-in member. Only call after the member has confirmed."""
        try:
            response = http.post("/bookings", json={
                "memberId": member_id, "teeSlotId": tee_slot_id, "players": players, "guests": guests})
        except httpx.HTTPError:
            return {"error": "The tee-time system is not available right now."}
        return response.json()  # the booking, or {"error": "<which club rule was broken>"}

    return [find_tee_times, book_tee_time] if member_id else [find_tee_times]


def _norm(s):
    return s.replace("–", "-").lower()


def answer(question, index, llm, k=TOP_K, member=True, tools=None, today=None):
    """Retrieve the top-k chunks the user may see, ask the LLM (optionally with tools), return {answer, sources}."""
    hits = index.search(question, k, member)
    context = "\n\n".join(f"[{c['doc']} – {c['section']}]\n{c['text']}" for c in hits)
    system = SYSTEM_PROMPT + (TOOLS_PROMPT if tools else "")
    user = (f"Today is {today}.\n\n" if today else "") + f"Context:\n{context}\n\nQuestion: {question}"
    text = llm(system, user, tools=tools).strip()
    if text.strip("'\" ").startswith(REFUSAL.rstrip(".")):
        return {"answer": REFUSAL, "sources": []}
    cited = [f"{c['doc']} – {c['section']}" for c in hits if _norm(f"[{c['doc']} – {c['section']}]") in _norm(text)]
    return {"answer": text, "sources": list(dict.fromkeys(cited))}
