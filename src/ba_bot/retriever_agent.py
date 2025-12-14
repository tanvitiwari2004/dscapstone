from typing import Any, Dict, List, Union

# If your Retriever lives in the same ba_bot package, use relative import:
from .retriever import Retriever # type: ignore


class RetrieverAgent:
    def __init__(self, top_k: int = 5):
        self.retriever = Retriever()
        self.top_k = top_k

    def retrieve(self, subqueries: Union[List[str], str]) -> List[Dict[str, Any]]:
        # Normalize input
        if isinstance(subqueries, str):
            subqueries = [subqueries]
        if not subqueries:
            return []

        all_hits: List[Dict[str, Any]] = []
        seen = set()

        for q in subqueries:
            if not isinstance(q, str):
                continue
            q = q.strip()
            if not q:
                continue

            hits = self.retriever.search(q, k=self.top_k) or []
            for h in hits:
                if not isinstance(h, dict):
                    continue

                cid = h.get("chunk_id")
                if not cid or cid in seen:
                    continue

                seen.add(cid)

                # Ensure fields exist to avoid sort crashes
                h.setdefault("score", 0.0)
                h.setdefault("text", "")

                all_hits.append(h)

        # Sort by score (descending) safely
        all_hits.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return all_hits
