"""Claude provider for the LLM forecasting experiment (Anthropic Python SDK)."""
import json

import anthropic

MODEL = "claude-opus-5-5"


class ClaudeForecaster:
    name = "claude"

    def __init__(self, model: str = MODEL):
        self.model = model
        self.client = anthropic.Anthropic()   # ANTHROPIC_API_KEY, or an `ant auth login` profile

    def forecast(self, system: str, prompt: str, schema: dict) -> dict:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",   # if a safety classifier declines, the API retries on a fallback model
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"refused: {response.stop_details}")
        text = next(b.text for b in response.content if b.type == "text")
        return {
            "parsed": json.loads(text),
            "raw": text,
            "model_returned": response.model,
            "stop_reason": response.stop_reason,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "request_id": response._request_id,
        }
