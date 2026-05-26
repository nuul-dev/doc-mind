import io
import uuid

from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.embedding import get_model
from app.types import IngestResult

COLLECTION = "docmind"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_DIM = 384  # all-MiniLM-L6-v2


def _ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )


def _parse_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _chunk_text(text: str) -> list[str]:
    words: list[str] = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[start:end]))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


def ingest_pdf(client: QdrantClient, filename: str, data: bytes) -> IngestResult:
    _ensure_collection(client)
    model = get_model()

    text = _parse_pdf(data)
    chunks = _chunk_text(text)
    embeddings = model.encode(chunks, batch_size=32, show_progress_bar=False, convert_to_numpy=True)  # pyright: ignore[reportUnknownMemberType]

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={"filename": filename, "chunk_index": i, "text": chunk},
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    client.upsert(collection_name=COLLECTION, points=points)

    return IngestResult(filename=filename, chunks=len(chunks))
