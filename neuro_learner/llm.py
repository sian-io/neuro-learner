import json
import re
from typing import Type, TypeVar
from pydantic import BaseModel
from google import genai
from google.genai import types

from neuro_learner.config import settings

T = TypeVar("T", bound=BaseModel)

class LLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None):
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model
        self.base_url = base_url or settings.llm_base_url
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["http_options"] = {"base_url": self.base_url}
            self._client = genai.Client(**kwargs)
        return self._client

    def generate_text(self, prompt: str, system_instruction: str | None = None) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return response.text.strip() if response.text else ""

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_instruction: str | None = None,
    ) -> T:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            raw_text = response.text or "{}"
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned.strip())
            return schema.model_validate_json(cleaned)
        except Exception:
            raw = self.generate_text(
                f"{prompt}\n\nStrictly output valid JSON matching this schema: {json.dumps(schema.model_json_schema())}",
                system_instruction=system_instruction,
            )
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned.strip())
            return schema.model_validate_json(cleaned)

default_llm_client = LLMClient()
