class LLMClient:
    def __init__(self, model="llama3.2:3b"):
        import ollama
        self.ollama = ollama
        self.model = model

    def chat(self, system: str, user: str) -> str:
        resp = self.ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": 0.2},
        )
        return resp["message"]["content"].strip()
