"""Tests for chunking, the /ask response shape and the refusal path.

They use a fake embedder and a fake LLM, so they run offline without an API key.
"""
import numpy as np

import rag

WORDS = ["guest", "pool", "tee", "dinner", "valet", "lottery"]


def fake_embed(texts):
    """Bag-of-keywords vectors: enough to make retrieval deterministic in tests."""
    vectors = np.array([[t.lower().count(w) for w in WORDS] for t in texts], dtype=float) + 1e-9
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_parse_doc_keeps_title_and_sections():
    title, sections = rag.parse_doc("# Pool\n\n## Hours\nOpen 10 to 8.\n\n## Rules\nNo glass.\n")
    assert title == "Pool"
    assert sections == [("Hours", "Open 10 to 8."), ("Rules", "No glass.")]


def test_split_words_overlaps_long_text():
    text = " ".join(str(i) for i in range(900))
    chunks = rag.split_words(text, size=400, overlap=50)
    assert [len(c.split()) for c in chunks] == [400, 400, 200]
    assert chunks[1].split()[0] == "350"  # second chunk starts 50 words before the first ends


def test_split_words_short_text_is_one_chunk():
    assert rag.split_words("Open daily.") == ["Open daily."]


def test_all_club_docs_load_with_metadata():
    chunks = rag.load_chunks()
    assert len({c["doc"] for c in chunks}) == 16
    assert all(c["doc"] and c["section"] and c["text"] for c in chunks)
    assert {c["visibility"] for c in chunks} == {"public", "members"}


def test_visitors_only_search_public_content():
    index = rag.Index(rag.load_chunks(), fake_embed)
    assert all(c["visibility"] == "public" for c in index.search("guest pool tee lottery", k=10, member=False))


def test_members_can_search_members_only_content():
    index = rag.Index(rag.load_chunks(), fake_embed)
    member_hits = index.search("When is the lottery deadline?", k=10, member=True)
    visitor_hits = index.search("When is the lottery deadline?", k=10, member=False)
    assert any(c["visibility"] == "members" for c in member_hits)
    assert not any(c["visibility"] == "members" for c in visitor_hits)


def test_answer_returns_text_and_cited_sources():
    index = rag.Index(rag.load_chunks(), fake_embed)
    top = index.search("How many guests can I bring?")[0]
    source = f"{top['doc']} – {top['section']}"
    llm = lambda system, user: f"Up to two guests. [{source}]"
    result = rag.answer("How many guests can I bring?", index, llm)
    assert set(result) == {"answer", "sources"}
    assert result["sources"] == [source]


def test_citation_with_plain_hyphen_still_counts():
    index = rag.Index(rag.load_chunks(), fake_embed)
    top = index.search("valet")[0]
    llm = lambda system, user: f"Fridays and Saturdays. [{top['doc']} - {top['section']}]"
    assert rag.answer("valet", index, llm)["sources"] == [f"{top['doc']} – {top['section']}"]


def test_refusal_returns_no_sources():
    index = rag.Index(rag.load_chunks(), fake_embed)
    llm = lambda system, user: rag.REFUSAL
    assert rag.answer("What is the Wi-Fi password?", index, llm) == {"answer": rag.REFUSAL, "sources": []}


def test_prompt_contains_context_and_refusal_rule():
    index = rag.Index(rag.load_chunks(), fake_embed)
    seen = {}

    def llm(system, user):
        seen["system"], seen["user"] = system, user
        return rag.REFUSAL

    rag.answer("When is the lottery deadline?", index, llm)
    assert rag.REFUSAL in seen["system"]
    assert "Question: When is the lottery deadline?" in seen["user"]
    assert "[Tee-Time Booking – Weekend lottery]" in seen["user"]
