from typing import Literal, TypedDict


class Chunk(TypedDict):
    score: float
    text: str
    filename: str
    chunk_index: int


class IngestResult(TypedDict):
    filename: str
    chunks: int


class EvalResult(TypedDict):
    faithfulness: float
    relevance: float
    completeness: float
    verdict: Literal["PASS", "FAIL"]
    reasoning: str


class Source(TypedDict):
    filename: str
    chunk_index: int
    score: float


class QueryResponse(TypedDict):
    answer: str
    sources: list[Source]
    eval: EvalResult  # optional, присутствует только если evaluate=True
