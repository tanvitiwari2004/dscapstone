class ReasonerAgent:
    def draft(self, question: str, contexts):
        if not contexts:
            return "I couldn't find relevant policy text in the knowledge base."

        # Build a concise answer using the top contexts, not full dumps
        lines = []
        lines.append(f"**Question:** {question}\n")
        lines.append("**Answer (grounded in BA policy text):**")

        # Pull key points from top chunks (first ~250 chars each)
        for c in contexts[:3]:
            txt = c["text"].strip().replace("\n", " ")
            if len(txt) > 260:
                txt = txt[:260].rsplit(" ", 1)[0] + "…"
            lines.append(f"- {txt} [{c['chunk_id']}]")

        citations = [c["chunk_id"] for c in contexts[:5]]
        lines.append("\n**Citations:** " + ", ".join([f"[{x}]" for x in citations]))
        return "\n".join(lines)
