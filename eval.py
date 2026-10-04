"""Evaluate Club Concierge on eval/questions.json.

Reports retrieval hit rate, answer accuracy and refusal accuracy.
Usage:  python eval.py                  (needs ANTHROPIC_API_KEY)
        python eval.py --retrieval-only (no API calls)
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

import rag


def main():
    load_dotenv()
    retrieval_only = "--retrieval-only" in sys.argv
    questions = json.loads((Path(__file__).parent / "eval" / "questions.json").read_text(encoding="utf-8"))
    index = rag.Index(rag.load_chunks(), rag.sentence_transformer_embedder())
    llm = None if retrieval_only else rag.claude_llm()

    answerable = [q for q in questions if q["doc"]]
    unanswerable = [q for q in questions if not q["doc"]]
    hits = correct = refused = 0

    for q in answerable:
        found = q["doc"] in {c["doc"] for c in index.search(q["question"])}
        hits += found
        line = f"[{'hit ' if found else 'MISS'}] {q['question']}"
        if llm:
            result = rag.answer(q["question"], index, llm)
            ok = any(f.lower() in result["answer"].lower() for f in q["facts"])
            correct += ok
            line += f"\n       {'OK   ' if ok else 'WRONG'} {result['answer'][:120]}"
        print(line)

    if llm:
        for q in unanswerable:
            result = rag.answer(q["question"], index, llm)
            ok = result["answer"] == rag.REFUSAL
            refused += ok
            print(f"[{'refused' if ok else 'ANSWERED'}] {q['question']}\n       {result['answer'][:120]}")

    print(f"\nRetrieval hit rate (top-{rag.TOP_K}): {hits}/{len(answerable)} = {hits / len(answerable):.0%}")
    if llm:
        print(f"Answer accuracy:   {correct}/{len(answerable)} = {correct / len(answerable):.0%}")
        print(f"Refusal accuracy:  {refused}/{len(unanswerable)}")


if __name__ == "__main__":
    main()
