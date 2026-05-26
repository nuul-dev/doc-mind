from typing import cast

from anthropic.types import TextBlock

from app.claude_client import get_client
from app.types import Chunk

MODEL = "claude-sonnet-4-6"


def generate_answer(question: str, chunks: list[Chunk]) -> str:
    context = "\n\n---\n\n".join(
        f"[{c['filename']}, chunk {c['chunk_index']}]\n{c['text']}" for c in chunks
    )

    prompt = f"""You are a precise document Q&A assistant.
Answer the question using ONLY the provided context.
If the answer cannot be found in the context, say so explicitly.

<context>
{context}
</context>

Question: {question}"""

    message = get_client().messages.create(
        model=MODEL,
        max_tokens=1024,
        system="Answer strictly from the provided context. Be concise and accurate.",
        messages=[{"role": "user", "content": prompt}],
    )

    return cast(TextBlock, message.content[0]).text
