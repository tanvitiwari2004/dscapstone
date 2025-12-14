from src.retriever import Retriever

SYSTEM_RULES = """You are a British Airways baggage policy assistant.
Rules:
- Answer ONLY using the retrieved chunks.
- If the retrieved chunks don’t contain the needed info, say what you can confirm and what you cannot.
- Always include citations as chunk_ids like [ba_lr_008].
- Prefer clear bullets and mention exceptions/requirements (e.g., prescription/medical letter/authorisation).
"""

class AgenticBot:
    def __init__(self, top_k: int = 5):
        self.retriever = Retriever()
        self.top_k = top_k

    def answer(self, question: str) -> str:
        # "Planner": turn question into a retrieval query
        retrieval_query = f"Find policy rules, limits, exceptions, approvals for: {question}"
        ctx = self.retriever.search(retrieval_query, k=self.top_k)

        if not ctx:
            return "I couldn’t retrieve any relevant policy text from the knowledge base."

        # "Answerer": grounded response (no hallucination)
        # We’ll synthesize using the retrieved snippets, but keep it strictly sourced.
        citations = [c["chunk_id"] for c in ctx]

        # Basic synthesis: pick the most relevant chunks and extract key lines.
        # (Still non-LLM; we’ll add an LLM after this works.)
        key_points = []
        for c in ctx[:3]:
            text = c["text"].strip().replace("\n", " ")
            key_points.append(f"- {text} [{c['chunk_id']}]")

        response = []
        response.append(f"{SYSTEM_RULES}\n")
        response.append(f"**Question:** {question}\n")
        response.append("**Grounded answer (from retrieved policy text):**")
        response.extend(key_points)

        response.append("\n**Citations used:** " + ", ".join([f"[{cid}]" for cid in citations]))
        return "\n".join(response)
