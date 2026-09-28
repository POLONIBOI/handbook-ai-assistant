"""The no-API-key option: a small open model that runs on the CPU.

    LLM_PROVIDER=local uvicorn app.api:app

Used automatically when ANTHROPIC_API_KEY isn't set. The model (Qwen3-0.6B,
about 1.5 GB) is downloaded from Hugging Face on first use and cached.

A 0.6B model gets plenty wrong (TEST_RESULTS.md has the details), so it gets
a simpler job than Claude and a safety net:

- It only answers or says NOT FOUND. Asked to also cite page numbers, a
  model this size starts writing the instructions back into its answers.
  The page is worked out afterwards instead: whichever excerpt shares the
  most distinctive words (figures, emails, names) with the answer.
- It gets the top 3 reranked excerpts, not 5. A longer prompt only gives it
  more to get confused by.
- The question is asked before and after the excerpts, so it reads them
  with the question in mind and still has it fresh when it answers.
- It isn't asked at all when the reranker's best score is below 0, i.e. no
  excerpt looks like an answer. Asked anyway, it answers "What is the pass
  mark for the MCQs?" with a made-up "15%" (the MCQ *weighting* on page 14).
"""

import re

from app import config
from app.llm import LLMAnswer, LLMError
from app.retriever import RetrievedChunk

SYSTEM_PROMPT = ("You answer questions from students on Zaio's Full-Stack AI Engineer Bootcamp, using only the "
                 "handbook excerpts you are given. You never use outside knowledge.")

INSTRUCTIONS = ("Answer the question in one to three sentences, using only the excerpts. Answer exactly "
                "what was asked. If the excerpts do not contain the answer, reply with exactly: NOT FOUND")

NOT_FOUND = re.compile(r"\bNOT FOUND\b", re.I)
# Words the model copies from the excerpts. Tokens with a digit (R 38,950,
# 70%, asif@zaio.io) say the most about where an answer came from.
TOKEN = re.compile(r"[a-z0-9@][a-z0-9@.,/'-]*[a-z0-9]", re.I)
COMMON = {"about", "after", "also", "answer", "before", "bootcamp", "excerpt", "excerpts", "from", "handbook",
          "have", "into", "need", "only", "that", "their", "there", "they", "this", "what", "when", "which",
          "will", "with", "your", "zaio"}
# Bits of the prompt a small model likes to repeat back.
SELF_REFERENCE = re.compile(r"\s*\(?\b(?:as (?:stated|mentioned|shown) in|according to|see|from) excerpt \d+\)?",
                            re.I)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    excerpts = "\n\n".join(f"Excerpt {i} (handbook page {c.page}, {c.section}):\n{c.text}"
                           for i, c in enumerate(chunks, start=1))
    user = f"Question: {question}\n\n{excerpts}\n\nQuestion: {question}\n\n{INSTRUCTIONS}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def distinctive_words(text: str) -> dict[str, int]:
    words = {}
    for token in TOKEN.findall(text.lower()):
        token = token.rstrip(".,")
        if any(ch.isdigit() for ch in token) or "@" in token:
            words[token] = 3
        elif len(token) >= 4 and token not in COMMON:
            words[token] = 1
    return words


def attribute(answer: str, chunks: list[RetrievedChunk]) -> int | None:
    """Index of the excerpt the answer most likely came from (None if none share a word).

    Only excerpts the reranker thinks answer the question (score 0 or more,
    and always the top one) can be cited. Without that, the first test run
    cited page 16 for a Capitec answer because page 16 mentions Capitec and
    Manati more often, although the reranker had scored it -2.24 against
    page 17's 5.26.
    """
    words = distinctive_words(answer)
    candidates = [i for i, c in enumerate(chunks)
                  if i == 0 or c.rerank_score is None or c.rerank_score >= config.LOCAL_MIN_RERANK_SCORE]
    scores = {i: sum(w for token, w in words.items() if token in distinctive_words(chunks[i].text))
              for i in candidates}
    best = max(scores.values(), default=0)
    if best == 0:
        return None
    return min(i for i, score in scores.items() if score == best)  # ties go to the higher-ranked excerpt


def parse_reply(reply: str, chunks: list[RetrievedChunk]) -> LLMAnswer:
    answer = reply.split("</think>")[-1]  # drop an (empty) think block if Qwen3 adds one
    if NOT_FOUND.search(answer) or not answer.strip():
        return LLMAnswer(found=False, answer="")
    answer = SELF_REFERENCE.sub("", answer)
    answer = re.sub(r"\s*[,;:]\s*(?=[.!?]|$)", "", answer)  # "7 months, as stated in Excerpt 1." left "7 months,."
    answer = re.sub(r"^(?:based on|according to) the excerpts,?\s*", "", answer.strip(), flags=re.I)
    answer = re.sub(r"\s+([.,;:!?])", r"\1", answer)
    answer = re.sub(r"\s+", " ", answer).strip()
    if answer:
        answer = answer[0].upper() + answer[1:]
    best = attribute(answer, chunks)
    return LLMAnswer(found=True, answer=answer, pages=[] if best is None else [chunks[best].page])


class LocalAnswerer:
    name = "local"

    def __init__(self, model: str = config.LOCAL_MODEL, generate=None):
        # `generate` (messages -> reply text) can be swapped out in tests so
        # they don't have to load the model.
        self.model = model
        self._generate = generate or self._load(model)

    @staticmethod
    def _load(name: str):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
        except Exception as exc:  # no network on first run, not enough memory, ...
            raise LLMError(f"Couldn't load the local model {name}: {exc}") from exc

        def generate(messages: list[dict]) -> str:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=config.LOCAL_MAX_NEW_TOKENS, do_sample=False)
            return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return generate

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> LLMAnswer:
        best = chunks[0].rerank_score if chunks else None
        if best is not None and best < config.LOCAL_MIN_RERANK_SCORE:
            return LLMAnswer(found=False, answer="")
        chunks = chunks[:config.LOCAL_MAX_EXCERPTS]
        try:
            reply = self._generate(build_messages(question, chunks))
        except Exception as exc:
            raise LLMError(f"Local model failed: {exc}") from exc
        return parse_reply(reply, chunks)
