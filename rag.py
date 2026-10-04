"""Club Concierge RAG pipeline: load -> chunk -> embed -> retrieve -> answer with citations."""
from pathlib import Path

import numpy as np

DOCS_DIR = Path(__file__).parent / "docs"
REFUSAL = "That isn't in the club's documents."
MODEL = "claude-opus-5-5"
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
    chunks = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        title, sections = parse_doc(path.read_text(encoding="utf-8"))
        for section, text in sections:
            for piece in split_words(text):
                chunks.append({"doc": title, "section": section, "text": piece})
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

    def search(self, question, k=TOP_K):
        scores = self.vectors @ np.asarray(self.embed([question]))[0]
        return [self.chunks[i] | {"score": float(scores[i])} for i in np.argsort(-scores)[:k]]


def claude_llm():
    """Returns a function (system, user) -> answer text, backed by the Claude API."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    def call(system, user):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # if a safety classifier declines, retry on a fallback model
        )
        if response.stop_reason == "refusal":
            return REFUSAL
        return "".join(block.text for block in response.content if block.type == "text")

    return call


def _norm(s):
    return s.replace("–", "-").lower()


def answer(question, index, llm, k=TOP_K):
    """Retrieve the top-k chunks, ask the LLM, and return {answer, sources}."""
    hits = index.search(question, k)
    context = "\n\n".join(f"[{c['doc']} – {c['section']}]\n{c['text']}" for c in hits)
    text = llm(SYSTEM_PROMPT, f"Context:\n{context}\n\nQuestion: {question}").strip()
    if text.strip("'\" ").startswith(REFUSAL.rstrip(".")):
        return {"answer": REFUSAL, "sources": []}
    cited = [f"{c['doc']} – {c['section']}" for c in hits if _norm(f"[{c['doc']} – {c['section']}]") in _norm(text)]
    return {"answer": text, "sources": list(dict.fromkeys(cited))}
