import json
from typing import List, Tuple


class EvaluatorAgent:
    """
    Generic evaluator:
    - Checks if the answer is incomplete, ambiguous, or overly cautious
    - Decides whether more retrieval is needed
    """

    def __init__(self, llm_client, max_extra: int = 4):
        self.llm = llm_client
        self.max_extra = max_extra
        self.system = """
You are an evaluator for a retrieval-grounded assistant.

Return ONLY valid JSON in this schema:
{
  "needs_more_evidence": true/false,
  "extra_queries": ["...", "..."],
  "reason": "short reason"
}

Rules:
- If the answer is incomplete, vague, or conditional without explanation,
  set needs_more_evidence=true.
- If the answer reasonably addresses the question using available context,
  set needs_more_evidence=false.
- extra_queries should help retrieve missing or clarifying information.
- JSON only. No commentary.
""".strip()

    def evaluate(
        self,
        question: str,
        user_context: str,
        answer: str,
        context_chunk_ids: List[str],
    ) -> Tuple[bool, List[str], str]:

        prompt = f"""
USER_CONTEXT:
{user_context}

QUESTION:
{question}

ANSWER:
{answer}
""".strip()

        raw = self.llm.chat(self.system, prompt)

        try:
            data = json.loads(raw)
            needs = bool(data.get("needs_more_evidence", False))
            extra = data.get("extra_queries", []) or []
            reason = str(data.get("reason", "")).strip()

            extra = [q.strip() for q in extra if isinstance(q, str) and q.strip()]
            extra = extra[: self.max_extra]

            return needs, extra if needs else [], reason

        except Exception:
            # Safe fallback: do not loop endlessly
            return False, [], "Evaluator output not parseable."
