import re
from memory import MemoryStore
from llm_client import LLMClient
from retriever_agent import RetrieverAgent

'''
from planner_agent import PlannerAgent
from reasoner_agent import ReasonerAgent
from evaluator_agent import EvaluatorAgent
from fact_extractor_agent import FactExtractorAgent
'''

SYSTEM_PROMPT = """
You are a helpful assistant.
Use ONLY the provided CONTEXT to answer.

Rules:
- Every sentence or bullet MUST end with at least one citation like [chunk_id].
- If a sentence cannot be supported by the CONTEXT, do not include it.
- If rules depend on route/country and USER_CONTEXT is missing details,
  ask ONE clarifying question and state what you can and cannot confirm.
""".strip()


def is_context_only(text: str) -> bool:
    """Heuristic: statement that sets context, not a policy question."""
    t = text.strip().lower()
    if "?" in t:
        return False

    patterns = [
        r"\bi(?:'| a)?m flying from\b",
        r"\bi(?:'| a)?m flying to\b",
        r"\bi(?:'| a)?m pregnant\b",
        r"\bi have\b.*\b(medical|cpap|oxygen|medication)\b",
        r"\bi am carrying\b",
    ]
    return any(re.search(p, t) for p in patterns)


def main():
    memory = MemoryStore(session_id="demo")
    llm = LLMClient(model="llama3.2:3b")
    retriever = RetrieverAgent(top_k=5)

    print("Generic RAG Chat (type 'exit' to quit)\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        memory.add_turn("user", q)

        # Context-only turn (no retrieval)
        if is_context_only(q):
            memory.set_fact("user_context", q)
            msg = "Got it — I’ll keep that context in mind."
            memory.add_turn("assistant", msg)
            print("\nBot:\n")
            print(msg)
            print("\n" + "-" * 70 + "\n")
            continue

        # 1) Retrieve relevant chunks
        contexts = retriever.retrieve([q])

        # 2) Build context block
        context_block = "\n\n---\n\n".join(
            f"[{c['chunk_id']}]\n{c['text']}" for c in contexts
        )

        user_context = memory.get_facts().get("user_context", "")

        user_prompt = f"""
USER_CONTEXT:
{user_context}

CONTEXT:
{context_block}

QUESTION:
{q}

Answer using ONLY the CONTEXT and cite chunk_ids.
""".strip()

        # 3) LLM grounded answer
        answer = llm.chat(SYSTEM_PROMPT, user_prompt)

        # 4) Save assistant turn with citations
        used = [c["chunk_id"] for c in contexts]
        memory.add_turn("assistant", answer, citations=used)

        print("\nBot:\n")
        print(answer)
        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()
