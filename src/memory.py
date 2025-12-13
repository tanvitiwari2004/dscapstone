import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
MEM_DIR = ROOT / "data" / "memory"
MEM_DIR.mkdir(parents=True, exist_ok=True)


class MemoryStore:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.path = MEM_DIR / f"session_{session_id}.json"
        self.data = {"session_id": session_id, "created_at": None, "updated_at": None,
                     "turns": [], "facts": {}}
        self.load()

    def load(self):
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            now = datetime.utcnow().isoformat()
            self.data["created_at"] = now
            self.data["updated_at"] = now
            self.save()

    def save(self):
        self.data["updated_at"] = datetime.utcnow().isoformat()
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    def add_turn(self, role: str, text: str, citations=None):
        turn = {"role": role, "text": text}
        if citations:
            turn["citations"] = citations
        self.data["turns"].append(turn)
        self.save()

    def set_fact(self, key: str, value):
        self.data["facts"][key] = value
        self.save()

    def get_facts(self):
        return self.data.get("facts", {})

    def get_recent_turns(self, n: int = 6):
        return self.data["turns"][-n:]
