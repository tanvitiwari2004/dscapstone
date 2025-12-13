import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

CHUNKS_PATH = Path("data/chunks.jsonl")
OUT_DIR = Path("data/index")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_chunks():
    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def main():
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    emb = np.array(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, str(OUT_DIR / "faiss.index"))
    (OUT_DIR / "chunk_meta.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    print(f"✅ Built index with {len(chunks)} chunks → {OUT_DIR}")

if __name__ == "__main__":
    main()
