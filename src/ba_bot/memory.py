import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

# Project root = ba-agentic-chatbot/
ROOT = Path(__file__).resolve().parents[2]
MEM_DIR = ROOT / "data" / "memory"
MEM_DIR.mkdir(parents=True, exist_ok=True)


class MemoryStore:
    """
    Simple JSON-backed session memory store.
    Safe for Streamlit reruns and partial writes.
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.path = MEM_DIR / f"session_{session_id}.json"
        self.data: Dict[str, Any] = {
            "session_id": session_id,
            "created_at": None,
            "updated_at": None,
            "turns": [],
            "facts": {},
        }
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                # Corrupt or partial file → reset safely
                self._init_new()
        else:
            self._init_new()

    def _init_new(self) -> None:
        now = datetime.utcnow().isoformat()
        self.data["created_at"] = now
        self.data["updated_at"] = now
        self.data["turns"] = []
        self.data["facts"] = {}
        self.save()

    def save(self) -> None:
        self.data["updated_at"] = datetime.utcnow().isoformat()
        self.path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def add_turn(
        self,
        role: str,
        text: str,
        citations: Optional[List[str]] = None,
    ) -> None:
        turn: Dict[str, Any] = {
            "role": role,
            "text": text,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if citations:
            turn["citations"] = citations

        self.data.setdefault("turns", []).append(turn)
        self.save()

    def set_fact(self, key: str, value: Any) -> None:
        self.data.setdefault("facts", {})[key] = value
        self.save()

    def get_facts(self) -> Dict[str, Any]:
        return self.data.get("facts", {})

    def get_recent_turns(self, n: int = 6) -> List[Dict[str, Any]]:
        return self.data.get("turns", [])[-n:]
