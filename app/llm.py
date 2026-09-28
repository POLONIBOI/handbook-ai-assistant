"""Part 2 (generation): write the answer from the retrieved chunks only.

Two answerers with the same interface, answer(question, chunks) -> LLMAnswer:

- ClaudeAnswerer: Claude through the Anthropic API. Used whenever an
  ANTHROPIC_API_KEY is set.
- LocalAnswerer (app/local_llm.py): a small open model on the CPU, so the
  project also runs, free, on a machine without a key.

make_answerer() picks one based on LLM_PROVIDER (see app/config.py).
"""

import json
import os
from dataclasses import dataclass, field

import anthropic

from app import config
from app.retriever import RetrievedChunk

SYSTEM_PROMPT = """\
You answer questions from students on Zaio's Full-Stack AI Engineer Bootcamp. \
You are given excerpts from the student handbook, each tagged with its page number.

Answer only from those excerpts. If they don't contain the answer, set "found" \
to false - don't fill the gap from general knowledge or from what bootcamps \
usually do, because a student may act on what you say. A related excerpt is \
not an answer: if someone asks for the MCQ pass mark and the handbook only \
gives the MCQ weighting, the answer is not in the handbook.

Write the way a tutor would reply on Discord: a short, direct answer, usually \
one to three sentences. Keep figures, times, links and email addresses exactly \
as the handbook writes them. In "pages", list only the pages of the excerpts \
you actually used."""

# Structured output, so the source page comes from the model's own reading of
# the excerpts rather than from whichever chunk happened to score highest.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "answer": {"type": "string"},
        "pages": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["found", "answer", "pages"],
    "additionalProperties": False,
}


class LLMError(RuntimeError):
    """Claude couldn't be reached or returned something unusable."""


class MissingAPIKeyError(LLMError):
    def __init__(self):
        super().__init__("ANTHROPIC_API_KEY is not set. Add it to .env, or set LLM_PROVIDER=local "
                         "to use the local model instead.")


@dataclass
class LLMAnswer:
    found: bool
    answer: str
    pages: list[int] = field(default_factory=list)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    excerpts = "\n\n".join(
        f'<excerpt page="{c.page}" section="{c.section}">\n{c.text}\n</excerpt>' for c in chunks
    )
    return f"Handbook excerpts:\n\n{excerpts}\n\nStudent question: {question}"


class ClaudeAnswerer:
    name = "claude"

    def __init__(self, client: anthropic.Anthropic | None = None, model: str = config.CLAUDE_MODEL):
        # Reads ANTHROPIC_API_KEY from the environment / .env
        self._client = client or anthropic.Anthropic()
        if self._client.api_key is None and self._client.auth_token is None:
            # Otherwise this only surfaces as a TypeError on the first request.
            raise MissingAPIKeyError()
        self._model = model

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> LLMAnswer:
        try:
            response = self._client.beta.messages.create(
                model=self._model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_prompt(question, chunks)}],
                # Short factual lookups don't need much thinking; low effort
                # keeps the API quick enough to sit behind an n8n webhook.
                output_config={
                    "effort": "low",
                    "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
                },
                # If the request is ever declined by a safety classifier, let the
                # API retry it on the recommended fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.APIError as exc:
            raise LLMError(f"Claude API request failed: {exc}") from exc

        if response.stop_reason == "refusal":
            return LLMAnswer(found=False, answer="")

        # Last text block, not first: if a fallback kicked in mid-answer, the
        # declined model's partial text comes before the fallback's answer.
        texts = [block.text for block in response.content if block.type == "text"]
        text = texts[-1] if texts else ""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Unexpected response from Claude (stop_reason={response.stop_reason})") from exc
        return LLMAnswer(found=data["found"], answer=data["answer"], pages=data["pages"])


def resolve_provider(provider: str = config.LLM_PROVIDER) -> str:
    """"claude", "local", or "auto" (Claude if a key is set, otherwise local)."""
    if provider == "auto":
        return "claude" if os.getenv("ANTHROPIC_API_KEY") else "local"
    if provider not in ("claude", "local"):
        raise ValueError(f"LLM_PROVIDER must be 'auto', 'claude' or 'local', not {provider!r}")
    return provider


def describe_llm(provider: str = config.LLM_PROVIDER) -> str:
    """For /health and the test report, e.g. "local: Qwen/Qwen3-0.6B"."""
    if resolve_provider(provider) == "claude":
        return f"claude: {config.CLAUDE_MODEL}"
    return f"local: {config.LOCAL_MODEL}"


def make_answerer(provider: str = config.LLM_PROVIDER):
    if resolve_provider(provider) == "claude":
        return ClaudeAnswerer()
    from app.local_llm import LocalAnswerer

    return LocalAnswerer()
