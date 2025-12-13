from memory import MemoryStore

class PlannerAgent:
    def plan(self, question: str, memory: MemoryStore):
        facts = memory.get_facts()
        hints = []

        # Use memory facts to shape retrieval (example)
        if "departure_country" in facts:
            hints.append(f"departure country: {facts['departure_country']}")

        # Produce 2–4 subqueries (this is your "plan")
        subqueries = [
            question,
            f"{question} limits exceptions approval requirements",
        ]
        if hints:
            subqueries.append(f"{question} ({', '.join(hints)})")

        return subqueries[:4]
