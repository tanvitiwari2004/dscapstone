import streamlit as st

from src.memory import MemoryStore
from src.llm_client import LLMClient
from src.retriever_agent import RetrieverAgent
from src.planner_agent import PlannerAgent
from src.evaluator_agent import EvaluatorAgent

SYSTEM_PROMPT = """
You are a helpful assistant.
Use the provided CONTEXT to answer the user's question.

Rules:
- Base your answer on the CONTEXT; do not invent facts.
- If multiple rules apply depending on route, country, or situation,
  explain the condition clearly.
- If essential details are missing, ask ONE clarifying question.
- Keep the answer concise and user-friendly.
""".strip()


def init_state():
    if "memory" not in st.session_state:
        st.session_state.memory = MemoryStore(session_id="ui_session")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "llm" not in st.session_state:
        st.session_state.llm = LLMClient(model="llama3.2:3b")
    if "retriever" not in st.session_state:
        st.session_state.retriever = RetrieverAgent(top_k=5)
    if "planner" not in st.session_state:
        st.session_state.planner = PlannerAgent(st.session_state.llm, max_subqueries=5)
    if "evaluator" not in st.session_state:
        st.session_state.evaluator = EvaluatorAgent(st.session_state.llm, max_extra=4)
    if "debug" not in st.session_state:
        st.session_state.debug = True


def build_context_block(contexts):
    return "\n\n---\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in contexts)


st.set_page_config(page_title="Agentic RAG Bot", page_icon="🧳", layout="centered")
init_state()

st.title("🧳 Agentic RAG Bot")
st.caption("Planner → Retriever → LLM → Evaluator → (optional) Re-retrieve")

with st.sidebar:
    st.subheader("Controls")
    st.session_state.debug = st.toggle("Show debug panels", value=st.session_state.debug)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.memory = MemoryStore(session_id="ui_session")
        st.rerun()

# Show existing conversation
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Ask a question…")
if q:
    mem = st.session_state.memory
    llm = st.session_state.llm
    retriever = st.session_state.retriever
    planner = st.session_state.planner
    evaluator = st.session_state.evaluator

    mem.add_turn("user", q)
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    user_context = mem.get_facts().get("user_context", "")

    # 1) PLAN
    subqueries = planner.plan(q, user_context=user_context)

    # 2) RETRIEVE
    contexts = retriever.retrieve(subqueries)
    used_ids = [c["chunk_id"] for c in contexts]
    context_block = build_context_block(contexts)

    # 3) DRAFT
    user_prompt = f"""
USER_CONTEXT:
{user_context}

SUBQUERIES:
{subqueries}

CONTEXT:
{context_block}

QUESTION:
{q}
""".strip()

    draft = llm.chat(SYSTEM_PROMPT, user_prompt)

    # 4) EVALUATE
    needs_more, extra_queries, reason = evaluator.evaluate(
        question=q, user_context=user_context, answer=draft, context_chunk_ids=used_ids
    )

    # 5) OPTIONAL RE-RETRIEVE ONCE
    if needs_more and extra_queries:
        contexts2 = retriever.retrieve(extra_queries)
        used_ids = [c["chunk_id"] for c in contexts2]
        context_block2 = build_context_block(contexts2)

        user_prompt2 = f"""
USER_CONTEXT:
{user_context}

EXTRA_QUERIES:
{extra_queries}

CONTEXT:
{context_block2}

QUESTION:
{q}
""".strip()

        answer = llm.chat(SYSTEM_PROMPT, user_prompt2)
    else:
        answer = draft

    mem.add_turn("assistant", answer, citations=used_ids)

    with st.chat_message("assistant"):
        st.markdown(answer)

        if st.session_state.debug:
            with st.expander("Debug: subqueries"):
                st.write(subqueries)
            with st.expander("Debug: chunk IDs"):
                st.write(used_ids)
            with st.expander("Debug: evaluator"):
                st.write({"needs_more": needs_more, "extra_queries": extra_queries, "reason": reason})

    st.session_state.messages.append({"role": "assistant", "content": answer})
