# Club Concierge AI Assistant

A retrieval-augmented generation (RAG) assistant that answers club members' questions **only from the club's approved documents**, cites the document and section it used, and replies *"That isn't in the club's documents."* when the answer isn't there.

> The club, **Maple Ridge Golf & Country Club**, is fictional. The 16 policy documents in `docs/` are sample data written for this project.

**Public vs. members-only content:** `docs/public/` is what the club's public website would show (dining, dress code, events, parking, contacts, app help). `docs/members/` is member-portal content (guest fees, tee-time and cancellation rules, billing, facilities). Visitors only get answers from public content; signed-in members get both.

## Agent tools: tee times

Besides answering from documents, the concierge can **act**. It has two tools that call the [Tee-Time Booking API](https://github.com/ridaali62/tee-time-booking):

| Tool | Who can use it | What it does |
|---|---|---|
| `find_tee_times(date, players)` | everyone | `GET /slots`, returns only slots with enough open places, plus the weather-delay flag |
| `book_tee_time(tee_slot_id, players, guests)` | signed-in members only | `POST /bookings` for that member. The API enforces the club rules (7-day window, 4 players, 2 guests) and returns the reason if a rule is broken |

The model decides when to call a tool. The Gemini SDK runs the loop (call tool → send result back → final answer). The prompt tells it to repeat the details and **only book after the member confirms**. Booking rules live in the booking API, not in the prompt, so the AI can't book something the club's rules don't allow.

Example: *"Any tee times this Saturday morning for 3 of us?"* → `find_tee_times("2026-10-10", 3)` → "07:10 and 07:30 are open…" → *"Book 07:10, one guest"* → confirms → `book_tee_time(2, 3, 1)`.

## Why

Club members ask the same questions all day: dining hours, guest rules, tee-time booking, cancellation fees. An assistant that answers instantly is only useful if members can trust it, so every answer must come from the club's own content and show its source. When the documents don't cover a question, it must say so instead of guessing.

## Architecture

```
docs/*.md ──► load + chunk ──► embed ──► NumPy vector index
                                              │
question ──► embed ──► top-k chunks (cosine) ─┘
                              │
                              ▼
            Gemini: "answer only from the context, cite [Doc – Section]"
                              │
                              ▼
                     {answer, sources[]}
```

| File | What it does |
|---|---|
| `rag.py` | Loading, chunking, embeddings, the vector index, retrieval and the grounded prompt |
| `app.py` | FastAPI app: `POST /ask`, `GET /health`, chat page at `/` |
| `rag.py` → `tee_time_tools` | The two agent tools that call the Tee-Time Booking API |
| `static/index.html` | Chat page (plain HTML + JavaScript) |
| `eval/questions.json` | 30 evaluation questions: 24 answerable, 6 not answerable from the documents |
| `eval.py` | Measures retrieval hit rate, answer accuracy and refusal accuracy |
| `tests/test_rag.py` | pytest tests for chunking, public/members-only filtering, the answer shape, the refusal path and the agent tools (offline, no API key; the booking API is mocked) |
| `.github/workflows/ci.yml` | Runs the tests on every push and pull request |

## Run it

Python 3.11+.

```
pip install -r requirements.txt
copy .env.example .env        # then put your free Gemini API key in .env (aistudio.google.com)
uvicorn app:app --reload
```

Open http://localhost:8000 and ask, e.g., *"Can I bring two guests on Saturday?"*. For tee-time questions, also start the [Tee-Time Booking API](https://github.com/ridaali62/tee-time-booking) on http://localhost:5080 (or set `TEE_TIME_API`).

API:

```
POST /ask   {"question": "When is the lottery deadline?", "member_id": 1}
→ {"answer": "Enter by Wednesday at 6:00 pm ... [Tee-Time Booking – Weekend lottery]",
   "sources": ["Tee-Time Booking – Weekend lottery"]}
GET /health → {"status": "ok", "chunks": 70}
```

Tests: `pytest`

## Evaluation

```
python eval.py                   # full run (calls the Gemini API)
python eval.py --retrieval-only  # retrieval only, no API key needed
```

- **Retrieval hit rate:** the expected document is among the top-k chunks
- **Answer accuracy:** the answer contains the key fact (e.g. "$110", "7 days")
- **Refusal accuracy:** the 6 out-of-scope questions get exactly "That isn't in the club's documents."

| Version | Change | Hit rate | Answer accuracy | Refusals |
|---|---|---|---|---|
| v1 | 400-word chunks, 50 overlap, k=4, `gemini-3.5-flash-lite` | 24/24 (100%) | 24/24 (100%) | 6/6 |

Answer accuracy is checked by key-fact matching (e.g. the answer must contain "$110"), which is lenient. A stricter check (a second model or human grading) is listed under next steps.

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
- **Model:** Google Gemini (`gemini-3.5-flash-lite`, free tier; its daily free allowance is higher than the larger Flash models') at temperature 0, so the same question gives the same answer and evaluation runs are repeatable. The model is set in one place (`GEMINI_MODEL`), and the pipeline only needs a function `(system, user) -> text`, so swapping in another provider (Claude, Azure OpenAI) is a one-function change. If the model returns no text (for example, a safety filter blocks it), the app replies with the refusal sentence instead of failing.

## Limitations and next steps

- Different member types (junior, social, full golf) should see different rules. Extend `visibility` to per-member-type permissions, taken from the signed-in session instead of a request flag.
- Let club staff upload and update documents through an admin page instead of editing files.
- For thousands of clubs, keep a separate document set and index per club, plus caching for frequent questions.
- Grade answers with a second model or by hand, instead of key-fact matching only.
