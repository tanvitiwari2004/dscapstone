import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Heavy imports inside init so the app can show a clear error if missing
# (and avoids slow import at module import time in Streamlit)


# repo root: .../ba-agentic-chatbot/
ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "data" / "index"


class Retriever:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: Path | None = None,
        meta_path: Path | None = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependency: sentence-transformers.\n"
                "Install with: python -m pip install sentence-transformers"
            ) from e

        try:
            import faiss  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependency: faiss.\n"
                "On Windows install: python -m pip install faiss-cpu\n"
                "If it fails, we can switch you to a scikit-learn fallback."
            ) from e

        self._faiss = faiss
        self.model = SentenceTransformer(model_name)

        index_path = index_path or (INDEX_DIR / "faiss.index")
        meta_path = meta_path or (INDEX_DIR / "chunk_meta.json")

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Chunk metadata not found at: {meta_path}")

        self.index = self._faiss.read_index(str(index_path))
        self.chunks: List[Dict[str, Any]] = json.loads(meta_path.read_text(encoding="utf-8"))

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        q = self.model.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")

        scores, idxs = self.index.search(q, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if int(idx) == -1:
                continue

            c = self.chunks[int(idx)]
            results.append(
                {
                    "chunk_id": c.get("chunk_id", f"chunk_{idx}"),
                    "section": c.get("section"),
                    "score": float(score),
                    "text": c.get("text", ""),
                    "source": c.get("source"),
                }
            )

        return results
