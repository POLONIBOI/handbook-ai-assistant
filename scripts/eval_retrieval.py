"""How often does retrieval find the right page? No LLM involved, so it's free to run.

    python scripts/eval_retrieval.py

This is what I used to pick TOP_K and MIN_SIMILARITY in app/config.py, and to
check the reranker and the local model's safety check.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.retriever import Retriever  # noqa: E402

# Question -> page (in a PDF viewer) that answers it.
LABELLED = [
    ("How much does the bootcamp cost?", 16),
    ("How is my final grade calculated?", 14),
    ("When are the live classes?", 11),
    ("What laptop do I need?", 5),
    ("What happens if my Capitec loan is declined?", 17),
    ("Who will be my tutor and can I change them?", 20),
    ("What are the tutor support hours?", 20),
    ("What mark do I need to get placed at a company?", 24),
    ("How much do the paid internships pay?", 24),
    ("What do the colours on my calendar mean?", 9),
    ("How do I check my XP rank on Discord?", 22),
    ("Where do I submit my assignments?", 13),
    ("When is orientation day?", 7),
    ("Who do I email about laptop advice?", 5),
    ("What name should I use on Discord?", 19),
    ("How do I join the Tuesday Zoom class?", 12),
    ("What are the payment options?", 16),
    ("Will the live classes be recorded?", 11),
    ("What will I learn in weeks 11 to 15?", 6),
    ("When is the deadline to pay fees?", 16),
    ("What should I do if Capitec hasn't contacted me?", 17),
    ("How do I get access to my calendar?", 10),
    ("Where are recorded lectures posted?", 13),
    ("What is Zaio's office address?", 26),
    ("How much do junior developers earn?", 24),
    ("How will you communicate with us before orientation?", 18),
    ("Which Discord levels get a certificate?", 22),
    ("Who is the lead instructor?", 15),
]

# Not in the handbook. Their best score shows why the threshold alone can't
# decide what's answerable.
UNANSWERABLE = [
    "What is the attendance requirement?",
    "What is the refund policy?",
    "Can I get an extension on an assignment?",
    "Is there a dress code for live classes?",
    "What is the pass mark for the MCQs?",
    "What is the Wi-Fi password at the Cape Town hub?",
    "Can I bring my dog to the live classes?",
    "Does Zaio offer medical aid for students?",
    "Who won the 2026 Zaio hackathon?",
    "What is the student-to-tutor ratio?",
    "Can I defer my studies to next year?",
    "Is there a student discount on laptops?",
    "What is the capital of France?",
    "How do I cook pasta?",
]


def best_similarity(hits) -> float:
    # max, not hits[0]: after reranking the first hit isn't the most similar one
    return max(h.similarity for h in hits)


def local_model_skips(hits) -> bool:
    """Would the local model's reranker check stop this question? (see app/local_llm.py)"""
    return hits[0].rerank_score is not None and hits[0].rerank_score < config.LOCAL_MIN_RERANK_SCORE


def main():
    by_similarity = Retriever(rerank=False)
    reranked = Retriever()
    top1 = topk = rr_top1 = rr_top3 = local_skips = 0
    lowest = 1.0

    print(f"Top {config.TOP_K} pages per question, similarity order (similarity of best hit) -> reranked\n")
    for question, page in LABELLED:
        hits = by_similarity.search(question)
        pages = [h.page for h in hits]
        top1 += pages[0] == page
        topk += page in pages
        lowest = min(lowest, best_similarity(hits))

        rr_hits = reranked.search(question)
        rr_pages = [h.page for h in rr_hits]
        rr_top1 += rr_pages[0] == page
        rr_top3 += page in rr_pages[:3]
        local_skips += local_model_skips(rr_hits)

        mark = "ok  " if pages[0] == page else ("top5" if page in pages else "MISS")
        print(f"  {mark} {best_similarity(hits):.2f}  want p{page:<3} got {pages} -> {rr_pages}  {question}")

    print("\nUnanswerable (best similarity, best rerank score)")
    past_gate = local_stopped = 0
    for question in UNANSWERABLE:
        hits = reranked.search(question)
        similarity = best_similarity(hits)
        if similarity < config.MIN_SIMILARITY:
            gate = "blocked (similarity)"
        else:
            past_gate += 1
            local_stopped += local_model_skips(hits)
            gate = "local model: skipped" if local_model_skips(hits) else "-> LLM decides"
        print(f"  {similarity:.2f}  {hits[0].rerank_score:6.2f}  {gate:<21} {question}")

    n = len(LABELLED)
    print(f"\nRight page ranked first: {top1}/{n} by similarity, {rr_top1}/{n} reranked")
    print(f"Right page in the top 3 after reranking (what the local model sees): {rr_top3}/{n}")
    print(f"Right page in top {config.TOP_K}:     {topk}/{n}")
    print(f"Lowest best-hit similarity on an answerable question: {lowest:.2f} "
          f"(threshold is {config.MIN_SIMILARITY})")
    print(f"Local model's reranker check stops {local_stopped} of the {past_gate} unanswerable questions that get "
          f"past the similarity gate, and {local_skips} of the {n} answerable ones")


if __name__ == "__main__":
    main()
