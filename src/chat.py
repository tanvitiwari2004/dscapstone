import re
from ba_bot.memory import MemoryStore
from ba_bot.llm_client import LLMClient
from ba_bot.retriever_agent import RetrieverAgent
from ba_bot.planner_agent import PlannerAgent
from ba_bot.evaluator_agent import EvaluatorAgent

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

    # If the user is asking a question, treat it as a question turn, not pure context.
    # (Even if they add context inside.)
    if "?" in t:
        return False

    patterns = [
        r"\bi(?:'| a)?m flying from\b",
        r"\bi(?:'| a)?m flying to\b",
        r"\bi(?:'| a)?m pregnant\b",
        r"\bi have\b.*\b(medical|cpap|oxygen|medication)\b",
        r"\bi am carrying\b",
        r"\broute\s*:\b",
        r"\btraveling from\b",
        r"\btravelling from\b",
    ]
    return any(re.search(p, t) for p in patterns)


def build_context_block(contexts: list[dict]) -> str:
    return "\n\n---\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in contexts)


def dedupe_contexts(contexts: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for c in contexts:
        cid = c.get("chunk_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(c)
    return out


def extract_citations(text: str) -> list[str]:
    # Extract bracketed ids like [abc_123] and dedupe preserving order.
    found = re.findall(r"\[([^\[\]]+)\]", text)
    deduped = []
    seen = set()
    for x in found:
        x = x.strip()
        if x and x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def has_any_citation(text: str) -> bool:
    return bool(re.search(r"\[[^\[\]]+\]", text))


def build_user_prompt(
    question: str,
    user_context: str,
    context_block: str,
    subqueries: list[str] | None = None,
    extra_queries: list[str] | None = None,
) -> str:
    parts = []

    parts.append("USER_CONTEXT:")
    parts.append(user_context or "")

    if subqueries:
        parts.append("\nSUBQUERIES (used for retrieval):")
        parts.append(str(subqueries))

    if extra_queries:
        parts.append("\nEXTRA_QUERIES (requested by evaluator):")
        parts.append(str(extra_queries))

    parts.append("\nCONTEXT:")
    parts.append(context_block)

    parts.append("\nQUESTION:")
    parts.append(question)

    parts.append("\nAnswer using ONLY the CONTEXT and cite chunk_ids.")
    return "\n".join(parts).strip()


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

        # Context-only turn: store it, acknowledge with a special citation token.
        if is_context_only(q):
            memory.set_fact("user_context", q)
            msg = "Got it — I’ll keep that context in mind. [user_context]"
            memory.add_turn("assistant", msg, citations=["user_context"])
            print("\nBot:\n")
            print(msg)
            print("\n" + "-" * 70 + "\n")
            continue

        user_context = memory.get_facts().get("user_context", "")

        # 1) PLAN → subqueries
        subqueries = planner.plan(q, user_context=user_context) or []
        if isinstance(subqueries, str):
            subqueries = [subqueries]

        # 2) RETRIEVE initial contexts
        contexts = retriever.retrieve(subqueries) or []
        contexts = dedupe_contexts(contexts)

        # 3) DRAFT answer
        context_block = build_context_block(contexts)
        user_prompt = build_user_prompt(
            question=q,
            user_context=user_context,
            context_block=context_block,
            subqueries=subqueries,
        )
        draft = llm.chat(SYSTEM_PROMPT, user_prompt)

        # 4) EVALUATE → maybe retrieve extra, then answer with merged context
        retrieved_ids_initial = [c["chunk_id"] for c in contexts]
        needs_more, extra_queries, reason = evaluator.evaluate(
            question=q,
            user_context=user_context,
            answer=draft,
            context_chunk_ids=retrieved_ids_initial,
        )

        all_contexts = list(contexts)
        if needs_more and extra_queries:
            contexts2 = retriever.retrieve(extra_queries) or []
            all_contexts = dedupe_contexts(all_contexts + contexts2)

        final_context_block = build_context_block(all_contexts)
        final_prompt = build_user_prompt(
            question=q,
            user_context=user_context,
            context_block=final_context_block,
            subqueries=subqueries,
            extra_queries=extra_queries if (needs_more and extra_queries) else None,
        )

        final_answer = llm.chat(SYSTEM_PROMPT, final_prompt)

        # Validate citations; reprompt once if the model forgot
        if not has_any_citation(final_answer):
            final_answer = llm.chat(
                SYSTEM_PROMPT,
                final_prompt
                + "\n\nIMPORTANT: Every sentence/bullet MUST end with at least one citation like [chunk_id].",
            )

        used_citations = extract_citations(final_answer)
        memory.add_turn("assistant", final_answer, citations=used_citations)

        print("\nBot:\n")
        print(final_answer)
        # Debug (optional)
        # print(f"\n[debug] evaluator: needs_more={needs_more} reason={reason} extra={extra_queries}\n")
        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()
