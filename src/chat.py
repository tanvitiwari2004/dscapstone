from memory import MemoryStore
from planner_agent import PlannerAgent
from retriever_agent import RetrieverAgent
from reasoner_agent import ReasonerAgent
from evaluator_agent import EvaluatorAgent

def main():
    memory = MemoryStore(session_id="demo")  # you can change session id later
    planner = PlannerAgent()
    retriever = RetrieverAgent(top_k=5)
    reasoner = ReasonerAgent()
    evaluator = EvaluatorAgent()

    print("BA Agentic RAG Bot (type 'exit' to quit)\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        memory.add_turn("user", q)

        subqueries = planner.plan(q, memory)
        contexts = retriever.retrieve(subqueries)
        draft = reasoner.draft(q, contexts)

        ok, extra_queries = evaluator.evaluate(q, draft)

# If evaluator asks for more evidence, do one more retrieval pass
        if extra_queries:
            contexts2 = retriever.retrieve(extra_queries)
            # merge (dedupe already happens in RetrieverAgent)
            contexts = contexts2
            draft = reasoner.draft(q, contexts)


        # save assistant turn + citations
        used = [c["chunk_id"] for c in contexts[:5]]
        memory.add_turn("assistant", draft, citations=used)

        print("\nBot:\n")
        print(draft)
        print("\n" + "-" * 70 + "\n")

if __name__ == "__main__":
    main()
