# ollama_client.py

import requests
import json

OLLAMA_BASE_URL = "http://localhost:11434"


def stream_chat(messages, model="llama3.1:8b"):
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": True
        },
        stream=True
    )

    for line in response.iter_lines():
        if line:
            decoded = json.loads(line.decode("utf-8"))
            yield decoded.get("message", {}).get("content", "")