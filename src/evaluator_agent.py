import re

class EvaluatorAgent:
    """
    Checks if the draft includes key requirements/exceptions that are commonly missed.
    If missing, returns extra retrieval prompts to fetch better evidence.
    """

    def evaluate(self, question: str, draft: str):
        cited = re.findall(r"\[ba_lr_\d+\]", draft)
        ok = len(cited) > 0

        extra_queries = []

        q = question.lower()
        d = draft.lower()

        # If user asked about medication/liquids over 100ml, ensure key terms appear
        if "medication" in q or "medicine" in q or "medical" in q:
            if "prescription" not in d and "medical letter" not in d:
                extra_queries.append("liquid medication hand baggage prescription medical letter beyond 100ml")
            if "original packaging" not in d:
                extra_queries.append("medication original packaging hand baggage requirement")
            if "security" not in d:
                extra_queries.append("avoid delays at security customs doctor letter medication")
        
        # General: if no citations, force retrieval
        if not ok:
            extra_queries.append("Find the exact BA policy text relevant to the question")

        return ok, extra_queries
