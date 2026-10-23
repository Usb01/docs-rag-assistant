import os

import httpx
import numpy as np


class OpenAIEmbeddings:
    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), 96):  # stay well under the request size limit
            resp = httpx.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": texts[i:i + 96]},
                timeout=60,
            )
            resp.raise_for_status()
            vectors.extend(item["embedding"] for item in sorted(resp.json()["data"], key=lambda d: d["index"]))
        arr = np.asarray(vectors, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def from_env():
    """Embeddings are optional. Without a key the index is keyword-only (BM25)."""
    if os.getenv("OPENAI_API_KEY") and os.getenv("EMBEDDINGS", "on") != "off":
        return OpenAIEmbeddings()
    return None
