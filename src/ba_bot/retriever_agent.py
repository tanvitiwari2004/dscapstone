from retriever import Retriever

class RetrieverAgent:
    def __init__(self, top_k: int = 5):
        self.retriever = Retriever()
        self.top_k = top_k

    def retrieve(self, subqueries):
        all_hits = []
        seen = set()

        for q in subqueries:
            hits = self.retriever.search(q, k=self.top_k)
            for h in hits:
                cid = h["chunk_id"]
                if cid not in seen:
                    seen.add(cid)
                    all_hits.append(h)

        # sort by score (descending)
        all_hits.sort(key=lambda x: x["score"], reverse=True)
        return all_hits
