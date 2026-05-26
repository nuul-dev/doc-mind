"""LLM-as-a-judge: evaluates faithfulness of the generated answer."""

import json
import re
from typing import cast

from anthropic.types import TextBlock

from app.claude_client import get_client
from app.generation import MODEL
from app.types import Chunk, EvalResult

EVAL_PROMPT = """You are an evaluation agent for a RAG system.
Given a question, retrieved context, and a generated answer, assess the answer quality.

Return a JSON object with these fields:
- faithfulness: float 0-1 (is the answer grounded in the context?)
- relevance: float 0-1 (does the answer address the question?)
- completeness: float 0-1 (does the answer cover all key points from context?)
- verdict: "PASS" | "FAIL"
- reasoning: brief explanation (1-2 sentences)

Rules:
- verdict is PASS if faithfulness >= 0.7 AND relevance >= 0.7
- Return ONLY valid JSON, no markdown fences

Question: {question}

Context:
{context}

Answer:
{answer}"""


def evaluate(question: str, chunks: list[Chunk], answer: str) -> EvalResult:
    context = "\n\n---\n\n".join(
        f"[{c['filename']}, chunk {c['chunk_index']}]\n{c['text']}" for c in chunks
    )

    prompt = EVAL_PROMPT.format(question=question, context=context, answer=answer)

    message = get_client().messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = cast(TextBlock, message.content[0]).text.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Claude sometimes wraps in backticks despite instructions
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Eval agent returned unparseable response: {raw!r}")
