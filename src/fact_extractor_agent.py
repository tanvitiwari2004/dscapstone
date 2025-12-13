import re
from memory import MemoryStore

class FactExtractorAgent:
    """
    Light-weight rule-based extractor (fast + reliable for demo).
    Extracts stable travel-context facts from user messages and stores them in memory.
    """

    def extract_and_store(self, user_text: str, memory: MemoryStore):
        txt = user_text.strip()

        # Example: "I'm flying from Australia" / "flying from Sydney"
        m = re.search(r"\bflying\s+from\s+([A-Za-z ]+?)(?:\s+and\b|,|\.|$)", txt, flags=re.I)

        if m:
            memory.set_fact("departure_place", m.group(1).strip())

        m = re.search(r"\bflying\s+to\s+([A-Za-z ]+?)(?:\s+and\b|,|\.|$)", txt, flags=re.I)
        if m:
            memory.set_fact("destination_place", m.group(1).strip())

        # Country keywords (extend later)
        if re.search(r"\bfrom\s+australia\b|\baustralia\b", txt, flags=re.I):
            memory.set_fact("departure_country", "Australia")

        if re.search(r"\bto\s+usa\b|\bunited states\b|\bamerica\b|\bto\s+the us\b", txt, flags=re.I):
            memory.set_fact("destination_country", "USA")

        # Pregnancy context
        if re.search(r"\bpregnan(t|cy)\b", txt, flags=re.I):
            memory.set_fact("traveller_pregnant", True)

        # Sports equipment intent
        if re.search(r"\bbike\b|\bbicycle\b|\bsurf(board)?\b|\bgolf\b|\bsk(i|is)\b|\bsnowboard\b|\bdiving\b|\bscuba\b|\bracket\b", txt, flags=re.I):
            memory.set_fact("topic_sports_equipment", True)

        # Medical intent
        if re.search(r"\bmedication\b|\bmedicine\b|\bmedical\b|\bcpap\b|\boxygen\b|\bdialysis\b", txt, flags=re.I):
            memory.set_fact("topic_medical", True)
