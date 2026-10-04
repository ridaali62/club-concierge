# Club Concierge AI Assistant

A retrieval-augmented generation (RAG) assistant that answers club members' questions **only from the club's approved documents**, cites the document and section it used, and replies *"That isn't in the club's documents."* when the answer isn't there.

> The club, **Maple Ridge Golf & Country Club**, is fictional. The 16 policy documents in `docs/` are sample data written for this project.

**Public vs. members-only content:** `docs/public/` is what the club's public website would show (dining, dress code, events, parking, contacts, app help). `docs/members/` is member-portal content (guest fees, tee-time and cancellation rules, billing, facilities). Visitors only get answers from public content; signed-in members get both.

## Why

Club members ask the same questions all day: dining hours, guest rules, tee-time booking, cancellation fees. An assistant that answers instantly is only useful if members can trust it, so every answer must come from the club's own content and show its source. When the documents don't cover a question, it must say so instead of guessing.

## Architecture

```
docs/*.md ──► load + chunk ──► embed ──► NumPy vector index
                                              │
question ──► embed ──► top-k chunks (cosine) ─┘
                              │
                              ▼
            Claude: "answer only from the context, cite [Doc – Section]"
                              │
                              ▼
                     {answer, sources[]}
```

| File | What it does |
|---|---|
| `rag.py` | Loading, chunking, embeddings, the vector index, retrieval and the grounded prompt |
| `app.py` | FastAPI app: `POST /ask`, `GET /health`, chat page at `/` |
| `static/index.html` | Chat page (plain HTML + JavaScript) |
| `eval/questions.json` | 30 evaluation questions: 24 answerable, 6 not answerable from the documents |
| `eval.py` | Measures retrieval hit rate, answer accuracy and refusal accuracy |
| `tests/test_rag.py` | pytest tests for chunking, public/members-only filtering, the answer shape and the refusal path (offline, no API key) |
| `.github/workflows/ci.yml` | Runs the tests on every push and pull request |

## Run it

Python 3.11+.

```
pip install -r requirements.txt
copy .env.example .env        # then put your Anthropic API key in .env
uvicorn app:app --reload
```

Open http://localhost:8000 and ask, e.g., *"Can I bring two guests on Saturday?"*

API:

```
POST /ask   {"question": "When is the lottery deadline?", "member": true}
→ {"answer": "Enter by Wednesday at 6:00 pm ... [Tee-Time Booking – Weekend lottery]",
   "sources": ["Tee-Time Booking – Weekend lottery"]}
GET /health → {"status": "ok", "chunks": 70}
```

Tests: `pytest`

## Evaluation

```
python eval.py                   # full run (calls the API)
python eval.py --retrieval-only  # retrieval only, no API key needed
```

- **Retrieval hit rate:** the expected document is among the top-k chunks
- **Answer accuracy:** the answer contains the key fact (e.g. "$110", "7 days")
- **Refusal accuracy:** the 6 out-of-scope questions get exactly "That isn't in the club's documents."

| Version | Change | Hit rate | Answer accuracy | Refusals |
|---|---|---|---|---|
| v1 | 400-word chunks, 50 overlap, k=4 | 24/24 (100%) | _not run yet_ | _not run yet_ |

## Design decisions

- **Access control at retrieval time:** every chunk carries `visibility`. The filter runs *before* the LLM sees anything, so members-only text can never leak into a visitor's answer through the prompt.
- **Chunking:** split by `##` section first, so each chunk keeps its `doc + section` metadata for citations. Sections longer than 400 words are split into 400-word windows with a 50-word overlap so a sentence on the boundary isn't lost. The club's sections are short, so most sections are one chunk.
- **Embeddings:** `all-MiniLM-L6-v2` from sentence-transformers runs locally and costs nothing. The chunk's title and section are embedded with the text, so "pool" questions match the Pool document even when a sentence doesn't say "pool".
- **Vector index:** a NumPy matrix. The vectors are normalised, so cosine similarity is a single dot product. For about 70 chunks, brute-force search is exact and instant. A vector database (FAISS, Chroma, pgvector) only pays off at a much larger scale.
- **k = 4:** enough context for questions that touch two sections (e.g. guest rules and guest fees), small enough to keep the prompt focused.
- **Preventing made-up answers:**
  1. The prompt says to answer only from the context and to reply with one exact sentence when the answer isn't there.
  2. Every answer must cite `[Doc – Section]`, and `sources` lists only sections that were actually retrieved *and* cited.
  3. The evaluation set includes questions the documents can't answer, so refusals are measured, not assumed.
- **Model:** Claude (`claude-opus-5-5`) at low effort: short factual answers don't need deep reasoning. Server-side fallback is on, so a request a safety classifier declines is retried on a fallback model.

## Limitations and next steps

- Different member types (junior, social, full golf) should see different rules. Extend `visibility` to per-member-type permissions, taken from the signed-in session instead of a request flag.
- Let club staff upload and update documents through an admin page instead of editing files.
- For thousands of clubs, keep a separate document set and index per club, plus caching for frequent questions.
- Grade answers with a second model or by hand, instead of key-fact matching only.
