from typing import cast

from qdrant_client import QdrantClient

from app.embedding import get_model
from app.ingest import COLLECTION
from app.types import Chunk

TOP_K = 5


def retrieve(client: QdrantClient, query: str, top_k: int = TOP_K) -> list[Chunk]:
    model = get_model()
    vector = model.encode(query, convert_to_numpy=True).tolist()  # pyright: ignore[reportUnknownMemberType]

    results = client.search(
        collection_name=COLLECTION,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
    )

    return [
        Chunk(
            score=hit.score,
            text=cast(str, hit.payload["text"]),
            filename=cast(str, hit.payload["filename"]),
            chunk_index=cast(int, hit.payload["chunk_index"]),
        )
        for hit in results
        if hit.payload is not None
    ]
