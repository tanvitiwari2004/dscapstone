import json
import re
from typing import List, Tuple, Any, Dict


class EvaluatorAgent:
    """
    Evaluator for a retrieval-grounded assistant.

    It decides if more retrieval is needed based on:
    - missing evidence
    - vague/conditional answers
    - ambiguity that could be resolved with more context

    Returns: (needs_more_evidence, extra_queries, reason)
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
- JSON only. No commentary. No markdown. No code fences.
""".strip()

    @staticmethod
    def _extract_json_object(text: str) -> str:
        """
        Some models wrap JSON in text or ```json fences.
        This extracts the first top-level JSON object substring.
        """
        if not text:
            return ""

        # Remove common code fences
        t = text.strip()
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)

        # Find first {...} block
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            return t[start : end + 1]
        return t

    @staticmethod
    def _safe_bool(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in {"true", "yes", "1", "y"}
        return False

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

AVAILABLE_CONTEXT_CHUNK_IDS:
{context_chunk_ids}

QUESTION:
{question}

ANSWER:
{answer}

Decide if more retrieval is needed. If yes, propose extra_queries that would retrieve missing evidence.
Return JSON only.
""".strip()

        raw = self.llm.chat(self.system, prompt)

        try:
            raw_json = self._extract_json_object(raw)
            data: Dict[str, Any] = json.loads(raw_json)

            # Allow mild key drift from the model
            needs = data.get("needs_more_evidence", data.get("needs_more", False))
            needs = self._safe_bool(needs)

            extra = data.get("extra_queries", data.get("queries", [])) or []
            reason = str(data.get("reason", data.get("why", "")) or "").strip()

            # Normalize extra queries
            if isinstance(extra, str):
                extra = [extra]
            if not isinstance(extra, list):
                extra = []

            extra_clean = []
            for q in extra:
                if isinstance(q, str):
                    q2 = q.strip()
                    if q2:
                        extra_clean.append(q2)

            extra_clean = extra_clean[: self.max_extra]

            return needs, (extra_clean if needs else []), reason

        except Exception:
            # Safe fallback: do not loop endlessly
            return False, [], "Evaluator output not parseable."
