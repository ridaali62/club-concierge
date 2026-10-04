"""Tests for chunking, the /ask response shape and the refusal path.

They use a fake embedder and a fake LLM, so they run offline without an API key.
"""
import httpx
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
    llm = lambda system, user, tools=None: f"Up to two guests. [{source}]"
    result = rag.answer("How many guests can I bring?", index, llm)
    assert set(result) == {"answer", "sources"}
    assert result["sources"] == [source]


def test_citation_with_plain_hyphen_still_counts():
    index = rag.Index(rag.load_chunks(), fake_embed)
    top = index.search("valet")[0]
    llm = lambda system, user, tools=None: f"Fridays and Saturdays. [{top['doc']} - {top['section']}]"
    assert rag.answer("valet", index, llm)["sources"] == [f"{top['doc']} – {top['section']}"]


def test_refusal_returns_no_sources():
    index = rag.Index(rag.load_chunks(), fake_embed)
    llm = lambda system, user, tools=None: rag.REFUSAL
    assert rag.answer("What is the Wi-Fi password?", index, llm) == {"answer": rag.REFUSAL, "sources": []}


def test_prompt_contains_context_and_refusal_rule():
    index = rag.Index(rag.load_chunks(), fake_embed)
    seen = {}

    def llm(system, user, tools=None):
        seen["system"], seen["user"] = system, user
        return rag.REFUSAL

    rag.answer("When is the lottery deadline?", index, llm)
    assert rag.REFUSAL in seen["system"]
    assert "Question: When is the lottery deadline?" in seen["user"]
    assert "[Tee-Time Booking – Weekend lottery]" in seen["user"]


SLOTS = [
    {"id": 1, "startTime": "2026-10-10T07:00:00", "capacity": 4, "bookedPlayers": 3, "weatherDelay": False},
    {"id": 2, "startTime": "2026-10-10T07:10:00", "capacity": 4, "bookedPlayers": 0, "weatherDelay": True},
]


def fake_api(handler):
    return httpx.Client(base_url="http://tee-time.test", transport=httpx.MockTransport(handler))


def test_find_tee_times_only_returns_slots_with_room():
    find = rag.tee_time_tools(http=fake_api(lambda request: httpx.Response(200, json=SLOTS)))[0]
    assert find("2026-10-10", 2) == [{"teeSlotId": 2, "time": "07:10", "openPlaces": 4, "weatherDelay": True}]


def test_visitors_can_only_look_up_members_can_also_book():
    assert [t.__name__ for t in rag.tee_time_tools(None, http=fake_api(None))] == ["find_tee_times"]
    assert [t.__name__ for t in rag.tee_time_tools(7, http=fake_api(None))] == ["find_tee_times", "book_tee_time"]


def test_book_tee_time_sends_member_and_returns_rule_errors():
    sent = {}

    def handler(request):
        sent["body"] = request.read()
        return httpx.Response(400, json={"error": "Only 1 place(s) left in this tee time."})

    book = rag.tee_time_tools(7, http=fake_api(handler))[1]
    assert book(1, 2, 0) == {"error": "Only 1 place(s) left in this tee time."}
    assert b'"memberId":7' in sent["body"].replace(b" ", b"")


def test_tools_report_when_booking_system_is_down():
    def handler(request):
        raise httpx.ConnectError("down")

    find = rag.tee_time_tools(http=fake_api(handler))[0]
    assert "error" in find("2026-10-10", 1)


def test_answer_passes_tools_and_date_to_llm():
    index = rag.Index(rag.load_chunks(), fake_embed)
    seen = {}

    def llm(system, user, tools=None):
        seen.update(system=system, user=user, tools=tools)
        return "07:10 is open."

    tools = rag.tee_time_tools(http=fake_api(None))
    rag.answer("Any tee times on Saturday?", index, llm, tools=tools, today="2026-10-08")
    assert seen["tools"] is tools
    assert "Today is 2026-10-08." in seen["user"]
    assert "only book after the member confirms" in seen["system"]
