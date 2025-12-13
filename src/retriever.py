import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root
INDEX_DIR = ROOT / "data" / "index"


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        self.chunks = json.loads((INDEX_DIR / "chunk_meta.json").read_text(encoding="utf-8"))

    def search(self, query: str, k: int = 5):
        q = self.model.encode([query], normalize_embeddings=True)
        q = np.array(q, dtype="float32")
        scores, idxs = self.index.search(q, k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            c = self.chunks[idx]
            results.append({
                "chunk_id": c["chunk_id"],
                "section": c.get("section"),
                "score": float(score),
                "text": c["text"],
                "source": c.get("source"),
            })
        return results
