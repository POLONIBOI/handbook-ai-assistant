"""The local model's prompt, reply parsing and page attribution, with a fake model (nothing is downloaded)."""

import pytest

from app import config
from app.llm import LLMError
from app.local_llm import LocalAnswerer, attribute, build_messages, parse_reply
from app.retriever import RetrievedChunk

CLASSES = RetrievedChunk(text="Live classes are on Tuesdays 9 am - 11 am and Thursdays 6 pm - 8 pm.",
                         page=11, section="Live Classes", similarity=0.7)
FEES = RetrievedChunk(text="Total Fees: R 38,950, payable before orientation day.", page=16, section="Fees",
                      similarity=0.5)
LAPTOP = RetrievedChunk(text="If you are looking to buy a laptop and are confused - please email asif@zaio.io.",
                        page=5, section="Hardware Requirements", similarity=0.4)
CHUNKS = [CLASSES, FEES, LAPTOP]


def test_question_comes_before_and_after_the_excerpts():
    system, user = build_messages("When are classes?", CHUNKS)
    content = user["content"]

    assert system["role"] == "system"
    assert content.startswith("Question: When are classes?")
    assert "Excerpt 1 (handbook page 11, Live Classes):\nLive classes are on" in content
    assert "Excerpt 3 (handbook page 5, Hardware Requirements)" in content
    assert content.index("Excerpt 3") < content.rindex("Question: When are classes?") < content.index("NOT FOUND")


@pytest.mark.parametrize("answer, expected", [
    ("Classes are on Tuesdays and Thursdays.", 0),
    ("The total fee is R 38,950.", 1),
    ("Email asif@zaio.io.", 2),
    ("Yes.", None),  # nothing distinctive to go on
])
def test_attribute_picks_the_excerpt_the_answer_came_from(answer, expected):
    assert attribute(answer, CHUNKS) == expected


def test_figures_outweigh_ordinary_words():
    # "fees" and "payable" match excerpt 1's wording here, but "38,950" only
    # appears in the fees excerpt.
    chunks = [RetrievedChunk(text="Fees are payable in instalments.", page=17, section="Fees", similarity=0.6), FEES]
    assert attribute("Fees are R 38,950.", chunks) == 1


def test_excerpts_the_reranker_rejected_are_never_cited():
    # From the first test run: the answer shares more words with page 16
    # ("Capitec", "Manati"), but the reranker scored page 16 as not answering.
    chunks = [
        RetrievedChunk(text="If you have not been contacted by Capitec, please email abi@zaio.io. Should your "
                            "Capitec loan fail, we will take your application to Manati.",
                       page=17, section="Fees", similarity=0.54, rerank_score=5.26),
        RetrievedChunk(text="Option 2: Financing with Capitec and Manati. Capitec offers education loans; Manati "
                            "offers student funding with flexible repayment.",
                       page=16, section="Fees", similarity=0.5, rerank_score=-2.24),
    ]
    answer = "Capitec will pass your application to Manati, which offers student funding and flexible repayment."
    assert attribute(answer, chunks) == 0


def test_parse_answer_gives_the_page():
    result = parse_reply("Classes are on Tuesdays and Thursdays.", CHUNKS)
    assert (result.found, result.answer, result.pages) == (True, "Classes are on Tuesdays and Thursdays.", [11])


def test_parse_drops_prompt_echoes_and_an_empty_think_block():
    reply = "<think>\n\n</think>\n\nBased on the excerpts, the total fee is R 38,950 (as stated in Excerpt 2)."
    result = parse_reply(reply, CHUNKS)
    assert result.answer == "The total fee is R 38,950."
    assert result.pages == [16]


def test_no_dangling_comma_after_removing_an_excerpt_reference():
    result = parse_reply("The total fee is R 38,950, as stated in Excerpt 2.", CHUNKS)
    assert result.answer == "The total fee is R 38,950."


@pytest.mark.parametrize("reply", [
    "NOT FOUND",
    "Not found.",
    "  ",
    "The excerpts don't mention a dress code. Therefore: NOT FOUND.",
    "<think>\n\n</think>\n\nNOT FOUND",
])
def test_parse_not_found(reply):
    assert parse_reply(reply, CHUNKS).found is False


def test_answer_uses_the_fake_model():
    seen = []

    def generate(messages):
        seen.append(messages)
        return "For laptop advice, email asif@zaio.io."

    result = LocalAnswerer(generate=generate).answer("Who do I email about laptops?", CHUNKS)
    assert (result.found, result.pages) == (True, [5])
    assert len(seen) == 1


def test_only_the_top_excerpts_are_sent(monkeypatch):
    monkeypatch.setattr(config, "LOCAL_MAX_EXCERPTS", 1)
    seen = []
    LocalAnswerer(generate=lambda m: seen.append(m) or "NOT FOUND").answer("q", CHUNKS)
    prompt = seen[0][1]["content"]
    assert "Live classes are on Tuesdays" in prompt
    assert "Excerpt 2" not in prompt


def test_model_is_not_asked_when_no_excerpt_fits():
    # A negative rerank score means the reranker doesn't think the excerpt
    # answers the question; a small model would answer anyway.
    weak = [RetrievedChunk(text="MCQs (15%): marked on Zaio platform.", page=14, section="Grade Breakdown",
                           similarity=0.47, rerank_score=-1.3)]
    calls = []
    result = LocalAnswerer(generate=lambda m: calls.append(m) or "The pass mark is 15%.").answer("MCQ pass mark?", weak)
    assert result.found is False
    assert calls == []


def test_model_is_asked_when_an_excerpt_fits():
    strong = [RetrievedChunk(text="Total Fees: R 38,950", page=16, section="Fees", similarity=0.6, rerank_score=5.1)]
    assert LocalAnswerer(generate=lambda m: "R 38,950.").answer("Fees?", strong).found is True


def test_model_failure_becomes_llm_error():
    def broken(messages):
        raise RuntimeError("out of memory")

    with pytest.raises(LLMError, match="out of memory"):
        LocalAnswerer(generate=broken).answer("q", CHUNKS)
