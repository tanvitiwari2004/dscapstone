from __future__ import annotations
from typing import Any, Dict, Optional


class LLMClient:
    def __init__(
        self,
        model: str = "llama3.2:3b",
        temperature: float = 0.2,
        timeout: int = 60,
    ):
        """
        Ollama client wrapper.

        model: Ollama model name (e.g., 'llama3.2:3b')
        temperature: decoding temperature
        timeout: seconds (best-effort; depends on ollama python client/version)
        """
        try:
            import ollama  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Failed to import 'ollama'. Install it with: python -m pip install ollama"
            ) from e

        self.ollama = ollama
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def chat(
        self,
        system: str,
        user: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        opts: Dict[str, Any] = {"temperature": self.temperature}
        if options:
            opts.update(options)

        try:
            resp = self.ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options=opts,
            )
        except Exception as e:
            # Give a useful message for the most common failure: Ollama not running
            raise RuntimeError(
                "Ollama request failed. Is the Ollama app/server running and the model pulled?\n"
                f"Model: {self.model}\n"
                "Try:\n"
                "  ollama serve\n"
                "  ollama pull llama3.2:3b"
            ) from e

        # Ollama Python responses can vary; handle safely
        msg = ""
        if isinstance(resp, dict):
            msg = (
                (resp.get("message") or {}).get("content")
                or resp.get("response")  # some clients use 'response'
                or ""
            )

        return str(msg).strip()
