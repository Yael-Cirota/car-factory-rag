"""Answers a question from the retrieved chunks, as a Pydantic object with citations.

Uses OpenAI. Needs OPENAI_API_KEY in the environment.
  LLM_MODEL=...     override the model (default: DEFAULT_MODEL)
"""
import os
from dataclasses import dataclass

from pydantic import BaseModel, Field

from rag.chunking import Chunk

DEFAULT_MODEL = "gpt-4o-mini"

# Dollars per million tokens (input, output). Check OpenAI's pricing page if you change models.
PRICES = {"gpt-4o-mini": (0.15, 0.60)}

SYSTEM = """אתה עונה על שאלות של לקוחות חברת הרכב "אופק רכב".
ענה רק על סמך הקטעים שבתוך <chunk>. אל תשתמש בידע כללי.
לכל טענה בתשובה, הוסף ציטוט: מזהה הקטע וציטוט מדויק מתוכו, מילה במילה.
אם הקטעים לא מספיקים כדי לענות, החזר insufficient_context=true, בלי ציטוטים.
ענה בעברית, במשפטים קצרים."""


class Citation(BaseModel):
    chunk_id: str = Field(description="The chunk id, exactly as it appears in <chunk id=...>")
    quote: str = Field(description="An exact quote from that chunk, word for word")


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    insufficient_context: bool


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


def build_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f'<chunk id="{c.id}">\n{c.text}\n</chunk>' for c in chunks)


def model_name() -> str:
    return os.environ.get("LLM_MODEL") or DEFAULT_MODEL


def ask(question: str, chunks: list[Chunk], client=None) -> tuple[Answer, Usage]:
    """Returns the parsed Answer and the token usage. Needs OPENAI_API_KEY in the environment."""
    if client is None:
        import openai
        client = openai.OpenAI()
    # Newer openai versions have chat.completions.parse; older ones only beta.chat.completions.parse.
    completions = client.chat.completions if hasattr(client.chat.completions, "parse") else client.beta.chat.completions
    response = completions.parse(
        model=model_name(),
        max_tokens=4000,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{build_context(chunks)}\n\nשאלה: {question}"},
        ],
        response_format=Answer,
    )
    message = response.choices[0].message
    if message.parsed is None:
        raise RuntimeError(f"no structured answer (refusal={message.refusal!r})")
    return message.parsed, Usage(response.usage.prompt_tokens, response.usage.completion_tokens)


def verify_citations(answer: Answer, chunks: list[Chunk]) -> list[Citation]:
    """Return the citations that are NOT backed by the chunks.

    A citation is bad if its chunk_id is not one of the chunks we sent,
    or if its quote does not appear in that chunk's text.
    An empty list means every citation checks out.
    """
    texts = {c.id: _normalize(c.text) for c in chunks}
    bad = []
    for cit in answer.citations:
        quote = _normalize(cit.quote)
        # An empty quote is "in" every string, so it proves nothing.
        if cit.chunk_id not in texts or not quote or quote not in texts[cit.chunk_id]:
            bad.append(cit)
    return bad


def _normalize(text: str) -> str:
    """Collapse every run of whitespace (spaces, tabs, newlines) into one space.

    A quote that spans a line break or a table row may come back with a space where the
    chunk has "\\n", or with doubled spaces. That is the same text, so it should pass.
    Everything else (letters, punctuation, numbers) must match exactly.
    """
    return " ".join(text.split())
