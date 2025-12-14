import json
from typing import List


class PlannerAgent:
    """
    Generic planner: creates 3–6 retrieval subqueries using the LLM.
    Output must be JSON: {"subqueries":[...]}
    """

    def __init__(self, llm_client, max_subqueries: int = 5):
        self.llm = llm_client
        self.max_subqueries = max_subqueries
        self.system = """
You are a retrieval planner.
Return ONLY valid JSON exactly in this schema:
{"subqueries": ["...", "..."]}

Rules:
- Create 3 to 6 subqueries that would help retrieve relevant policy text.
- Use USER_CONTEXT if provided to specialize (route/country/constraints).
- No extra keys. No commentary. JSON only.
""".strip()

    def plan(self, question: str, user_context: str = "") -> List[str]:
        prompt = f"""
USER_CONTEXT:
{user_context}

QUESTION:
{question}
""".strip()

        raw = self.llm.chat(self.system, prompt)

        # strict JSON parse
        try:
            data = json.loads(raw)
            sq = data.get("subqueries", [])
            sq = [s.strip() for s in sq if isinstance(s, str) and s.strip()]
            if sq:
                return sq[: self.max_subqueries]
        except Exception:
            pass

        # fallback if model outputs non-JSON
        return [question]
