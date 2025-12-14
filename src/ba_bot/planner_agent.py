import json
import re
from typing import List, Any, Dict


class PlannerAgent:
    """
    Generic planner: creates retrieval subqueries using the LLM.
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
- Keep subqueries short (max ~12 words each).
- Avoid duplicates.
- No extra keys. No commentary. JSON only. No markdown. No code fences.
""".strip()

    @staticmethod
    def _extract_json_object(text: str) -> str:
        """Extract the first {...} JSON object even if wrapped in text/fences."""
        if not text:
            return ""
        t = text.strip()
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            return t[start : end + 1]
        return t

    @staticmethod
    def _normalize_subqueries(items: Any) -> List[str]:
        if isinstance(items, str):
            items = [items]
        if not isinstance(items, list):
            return []

        out: List[str] = []
        seen = set()

        for s in items:
            if not isinstance(s, str):
                continue
            q = " ".join(s.strip().split())  # collapse whitespace
            if not q:
                continue
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(q)

        return out

    def plan(self, question: str, user_context: str = "") -> List[str]:
        prompt = f"""
USER_CONTEXT:
{user_context}

QUESTION:
{question}

Return JSON only.
""".strip()

        raw = self.llm.chat(self.system, prompt)

        try:
            raw_json = self._extract_json_object(raw)
            data: Dict[str, Any] = json.loads(raw_json)
            sq = self._normalize_subqueries(data.get("subqueries", []))
        except Exception:
            sq = []

        # Ensure we have enough queries (LLMs often return 1–2)
        if len(sq) < 3:
            # deterministic fallback expansion (no extra LLM call)
            base = question.strip()
            extras = [
                base,
                f"{base} policy",
                f"{base} restrictions",
                f"{base} allowed items",
                f"{base} carry-on vs checked baggage",
            ]
            if user_context.strip():
                extras.insert(1, f"{base} {user_context.strip()}")
            sq = self._normalize_subqueries(extras)

        return sq[: self.max_subqueries]
