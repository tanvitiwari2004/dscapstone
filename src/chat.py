import re
from src.memory import MemoryStore
from src.llm_client import LLMClient
from src.retriever_agent import RetrieverAgent
from src.planner_agent import PlannerAgent
from src.evaluator_agent import EvaluatorAgent

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


def build_context_block(contexts):
    return "\n\n---\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in contexts)


def main():
    memory = MemoryStore(session_id="demo")
    llm = LLMClient(model="llama3.2:3b")
    retriever = RetrieverAgent(top_k=5)
    planner = PlannerAgent(llm, max_subqueries=5)
    evaluator = EvaluatorAgent(llm, max_extra=4)

    print("Generic Agentic RAG Chat (type 'exit' to quit)\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        memory.add_turn("user", q)

        # context-only turn: acknowledge + store
        if is_context_only(q):
            memory.set_fact("user_context", q)
            msg = "Got it — I’ll keep that context in mind."
            memory.add_turn("assistant", msg)
            print("\nBot:\n")
            print(msg)
            print("\n" + "-" * 70 + "\n")
            continue

        user_context = memory.get_facts().get("user_context", "")

        # 1) PLAN → subqueries
        subqueries = planner.plan(q, user_context=user_context)

        # 2) RETRIEVE
        contexts = retriever.retrieve(subqueries)
        context_block = build_context_block(contexts)

        # 3) ANSWER (draft)
        user_prompt = f"""
USER_CONTEXT:
{user_context}

SUBQUERIES (used for retrieval):
{subqueries}

CONTEXT:
{context_block}

QUESTION:
{q}

Answer using ONLY the CONTEXT and cite chunk_ids.
""".strip()

        draft = llm.chat(SYSTEM_PROMPT, user_prompt)

        # 4) EVALUATE → maybe re-retrieve once
        used_ids = [c["chunk_id"] for c in contexts]
        needs_more, extra_queries, reason = evaluator.evaluate(
            question=q,
            user_context=user_context,
            answer=draft,
            context_chunk_ids=used_ids,
        )

        if needs_more and extra_queries:
            contexts2 = retriever.retrieve(extra_queries)
            context_block2 = build_context_block(contexts2)

            user_prompt2 = f"""
USER_CONTEXT:
{user_context}

EXTRA_QUERIES (requested by evaluator):
{extra_queries}

CONTEXT:
{context_block2}

QUESTION:
{q}

Answer using ONLY the CONTEXT and cite chunk_ids.
""".strip()

            final_answer = llm.chat(SYSTEM_PROMPT, user_prompt2)
            final_used_ids = [c["chunk_id"] for c in contexts2]
        else:
            final_answer = draft
            final_used_ids = used_ids

        memory.add_turn("assistant", final_answer, citations=final_used_ids)

        print("\nBot:\n")
        print(final_answer)
        # optional debug line (keep or remove)
        # print(f"\n[debug] evaluator: {needs_more=} reason={reason} extra={extra_queries}\n")
        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()
