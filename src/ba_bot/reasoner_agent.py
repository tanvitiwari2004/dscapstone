import re

class ReasonerAgent:
    def draft(self, question: str, contexts, memory_facts=None):
        if not contexts:
            return "I couldn't find relevant policy text in the knowledge base."

        memory_facts = memory_facts or {}
        q_lower = question.lower()

        # Keywords to fish out relevant sentences
        keywords = []
        if "powder" in q_lower:
            keywords += ["powder", "powders", "350g", "350ml"]
        if memory_facts.get("departure_country") == "Australia":
            keywords += ["australia", "inorganic"]

        # Sentence extraction helper
        def pick_best_snippets(text: str):
            # split loosely into sentences / bullet-like segments
            parts = re.split(r"(?<=[\.\!\?])\s+|\n+|•|\u2022|- ", text)
            parts = [p.strip() for p in parts if p.strip()]
            if not keywords:
                return parts[:2]
            hits = [p for p in parts if any(k in p.lower() for k in keywords)]
            return hits[:3] if hits else parts[:2]

        lines = []
        lines.append(f"**Question:** {question}\n")
        lines.append("**Answer (grounded in BA policy text):**")

        for c in contexts[:3]:
            snippets = pick_best_snippets(c["text"])
            joined = " ".join(snippets)
            joined = joined.replace("\n", " ").strip()
            if len(joined) > 280:
                joined = joined[:280].rsplit(" ", 1)[0] + "…"
            lines.append(f"- {joined} [{c['chunk_id']}]")

        citations = [c["chunk_id"] for c in contexts[:5]]
        lines.append("\n**Citations:** " + ", ".join([f"[{x}]" for x in citations]))
        return "\n".join(lines)
