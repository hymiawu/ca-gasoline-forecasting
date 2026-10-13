"""ChatGPT provider for the LLM forecasting experiment (OpenAI Python SDK, Responses API)."""
import json
import os

import openai

# Set OPENAI_MODEL to the model you want to test; the model that actually answered is logged.
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5")


class OpenAIForecaster:
    name = "openai"

    def __init__(self, model: str = MODEL):
        self.model = model
        self.client = openai.OpenAI()   # OPENAI_API_KEY

    def forecast(self, system: str, prompt: str, schema: dict) -> dict:
        response = self.client.responses.create(
            model=self.model,
            instructions=system,
            input=prompt,
            text={"format": {"type": "json_schema", "name": "forecast", "schema": schema, "strict": True}},
        )
        text = response.output_text
        usage = response.usage
        return {
            "parsed": json.loads(text),
            "raw": text,
            "model_returned": response.model,
            "stop_reason": response.status,
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "request_id": response.id,
        }
