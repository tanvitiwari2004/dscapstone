import re
from memory import MemoryStore
from llm_client import LLMClient

'''
from planner_agent import PlannerAgent
from retriever_agent import RetrieverAgent
from reasoner_agent import ReasonerAgent
from evaluator_agent import EvaluatorAgent
from fact_extractor_agent import FactExtractorAgent
'''

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

    print("Generic LLM Chat (type 'exit' to quit)\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        memory.add_turn("user", q)

        # If user is just setting context, acknowledge and continue
        if is_context_only(q):
            msg = "Got it — I’ll keep that context in mind."
            memory.add_turn("assistant", msg)
            print("\nBot:\n")
            print(msg)
            print("\n" + "-" * 70 + "\n")
            continue

        answer = llm.chat("You are a helpful assistant.", q)
        memory.add_turn("assistant", answer)

        print("\nBot:\n")
        print(answer)
        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()
