"""Club Concierge RAG pipeline: load -> chunk -> embed -> retrieve -> answer with citations."""
import os
from pathlib import Path

import numpy as np

DOCS_DIR = Path(__file__).parent / "docs"
REFUSAL = "That isn't in the club's documents."
MODEL = "gemini-2.5-flash"  # override with GEMINI_MODEL in .env
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

    client = genai.Client()  # reads GEMINI_API_KEY from the environment
    model = model or os.getenv("GEMINI_MODEL", MODEL)

    def call(system, user):
        response = client.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(system_instruction=system, temperature=0),
        )
        return response.text or REFUSAL  # empty text (e.g. blocked by a safety filter) -> refuse

    return call


def _norm(s):
    return s.replace("–", "-").lower()


def answer(question, index, llm, k=TOP_K, member=True):
    """Retrieve the top-k chunks the user may see, ask the LLM, and return {answer, sources}."""
    hits = index.search(question, k, member)
    context = "\n\n".join(f"[{c['doc']} – {c['section']}]\n{c['text']}" for c in hits)
    text = llm(SYSTEM_PROMPT, f"Context:\n{context}\n\nQuestion: {question}").strip()
    if text.strip("'\" ").startswith(REFUSAL.rstrip(".")):
        return {"answer": REFUSAL, "sources": []}
    cited = [f"{c['doc']} – {c['section']}" for c in hits if _norm(f"[{c['doc']} – {c['section']}]") in _norm(text)]
    return {"answer": text, "sources": list(dict.fromkeys(cited))}
