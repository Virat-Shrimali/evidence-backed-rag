"""Provider-agnostic LLM abstraction layer and provider implementations."""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from src.config import INSUFFICIENT_EVIDENCE_REFUSAL, RAGConfig, settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM generation providers."""

    @abstractmethod
    def generate_raw(self, prompt: str, system_prompt: str) -> str:
        """Generate raw text response from prompt and system instructions."""
        pass


class MockLLMProvider(BaseLLMProvider):
    """Deterministic mock provider for offline unit testing without API keys or network."""

    def __init__(
        self,
        canned_response: str | None = None,
        canned_responses: dict[str, str] | list[str] | None = None,
        generator_fn: Callable[[str, str], str] | None = None,
    ):
        self.canned_response = canned_response
        self.canned_responses = canned_responses or {}
        self.generator_fn = generator_fn
        self.call_history: list[dict[str, str]] = []
        self._list_index = 0

    def generate_raw(self, prompt: str, system_prompt: str) -> str:
        """Generate deterministic response or return configured canned response."""
        self.call_history.append({"prompt": prompt, "system_prompt": system_prompt})

        # 1. Custom generator function
        if self.generator_fn is not None:
            return self.generator_fn(prompt, system_prompt)

        # 2. Exact canned response
        if self.canned_response is not None:
            return self.canned_response

        # 3. List of sequential canned responses
        if isinstance(self.canned_responses, list) and self.canned_responses:
            resp = self.canned_responses[self._list_index % len(self.canned_responses)]
            self._list_index += 1
            return resp

        # 4. Dict mapping query keywords to responses
        if isinstance(self.canned_responses, dict):
            for key, val in self.canned_responses.items():
                if key.lower() in prompt.lower():
                    return val

        # 5. Default heuristic behavior: generate grounded response from context
        # Extract question from prompt using the final Question: section
        question_text = ""
        if "\nQuestion:" in prompt:
            question_part = prompt.rsplit("\nQuestion:", 1)[-1]
            question_text = question_part.split("\n\nRemember to respond")[0].strip()
        elif "Question:" in prompt:
            question_part = prompt.rsplit("Question:", 1)[-1]
            question_text = question_part.split("\n\nRemember to respond")[0].strip()
        else:
            question_text = prompt

        # Detect refusal / unanswerable trigger ONLY in question, NOT in context
        if any(w in question_text.lower() for w in ["unanswerable", "redis", "elasticsearch", "postgresql connection"]):
            return json.dumps({
                "answer": INSUFFICIENT_EVIDENCE_REFUSAL,
                "citations": [],
                "sufficient_evidence": False,
                "confidence": 0.0,
                "refusal_reason": "No evidence found in context",
            })

        # Extract context part between "Context:\n" and final "\n\nQuestion:"
        context_part = ""
        if "Context:\n" in prompt:
            context_part = prompt.split("Context:\n", 1)[-1]
            if "\n\nQuestion:" in context_part:
                context_part = context_part.rsplit("\n\nQuestion:", 1)[0]
            elif "\nQuestion:" in context_part:
                context_part = context_part.rsplit("\nQuestion:", 1)[0]
        else:
            context_part = prompt

        chunk_blocks = re.findall(
            r"--- Chunk ID: ([\w\-#]+) ---\n(.*?)(?=(?:--- Chunk ID:|\Z))",
            context_part,
            flags=re.DOTALL,
        )

        chosen_chunk_id = "mock_chunk_0"
        chosen_snippet = "Sample verified fact."

        if chunk_blocks:
            # Check if any chunk matches salient query terms from question
            q_terms = [
                w.lower()
                for w in re.findall(r"\b\w+\b", question_text)
                if len(w) > 3 and w.lower() not in {
                    "what", "which", "where", "when", "how", "why", "who", "does",
                    "the", "this", "that", "these", "those", "have", "with", "from",
                    "project", "using", "used", "tell", "show", "give",
                }
            ]

            best_chunk_id = None
            best_snippet = None
            best_score = (0, 0, 0)  # (best_line_terms, unique_terms, total_count)

            for cid, content in chunk_blocks:
                lines = [line.strip() for line in content.splitlines() if line.strip()]
                content_lower = content.lower()
                unique_terms = sum(1 for term in q_terms if term in content_lower)
                if unique_terms == 0:
                    continue
                total_count = sum(content_lower.count(term) for term in q_terms)

                chunk_best_line = lines[0] if lines else ""
                chunk_best_line_terms = 0
                for line in lines:
                    clean = line.lstrip("- *#│").strip()
                    clean_lower = clean.lower()
                    l_terms = sum(
                        1
                        for term in q_terms
                        if term in clean_lower or (len(term) > 4 and term.rstrip("s") in clean_lower)
                    )
                    if "model" in q_terms and any(
                        m in clean_lower for m in ["minilm", "sentence-transformers", "all-minilm", "bge-small"]
                    ):
                        l_terms += 3
                    if l_terms > chunk_best_line_terms:
                        chunk_best_line_terms = l_terms
                        chunk_best_line = clean

                score = (chunk_best_line_terms, unique_terms, total_count)
                if score > best_score:
                    best_score = score
                    best_chunk_id = cid
                    best_snippet = chunk_best_line[:120].strip()

            if best_chunk_id and best_snippet:
                chosen_chunk_id = best_chunk_id
                chosen_snippet = best_snippet
            else:
                first_cid, first_content = chunk_blocks[0]
                lines = [line.strip() for line in first_content.splitlines() if line.strip()]
                chosen_chunk_id = first_cid
                first_line = lines[0] if lines else "Sample verified fact."
                chosen_snippet = first_line[:60].strip()
        else:
            chunk_id_match = re.search(r"--- Chunk ID: ([\w\-#]+) ---", prompt)
            chosen_chunk_id = chunk_id_match.group(1) if chunk_id_match else "mock_chunk_0"
            content_lines = [
                line.strip()
                for line in context_part.splitlines()
                if line.strip() and not line.startswith("---")
            ]
            first_line = content_lines[0] if content_lines else "Sample verified fact."
            chosen_snippet = first_line[:60].strip()

        return json.dumps({
            "answer": f"Based on the provided documentation, {chosen_snippet}.",
            "citations": [{"chunk_id": chosen_chunk_id, "text_snippet": chosen_snippet}],
            "sufficient_evidence": True,
            "confidence": 0.95,
            "refusal_reason": None,
        })


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI API provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.0,
    ):
        self.api_key = api_key or settings.openai_api_key
        self.model_name = model_name or settings.llm_model_name
        self.temperature = temperature
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError as e:
                raise ImportError(
                    "The openai package is required to use OpenAILLMProvider. Install it with: pip install openai"
                ) from e
        return self._client

    def generate_raw(self, prompt: str, system_prompt: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""


class AnthropicLLMProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.0,
    ):
        self.api_key = api_key or settings.anthropic_api_key
        self.model_name = model_name or "claude-3-5-sonnet-20241022"
        self.temperature = temperature
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError as e:
                raise ImportError(
                    "The anthropic package is required to use AnthropicLLMProvider. Install it with: pip install anthropic"
                ) from e
        return self._client

    def generate_raw(self, prompt: str, system_prompt: str) -> str:
        client = self._get_client()
        response = client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=1024,
        )
        return response.content[0].text if response.content else ""


class OllamaLLMProvider(BaseLLMProvider):
    """Local Ollama provider via native HTTP requests (zero extra dependency)."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.0,
    ):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model_name = model_name or "llama3.1:8b"
        self.temperature = temperature

    def generate_raw(self, prompt: str, system_prompt: str) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
                return resp_json.get("message", {}).get("content", "")
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Failed to connect to local Ollama instance at {self.base_url}: {e}"
            ) from e


def create_llm_provider(
    provider_type: str | None = None,
    config: RAGConfig | None = None,
    **kwargs: Any,
) -> BaseLLMProvider:
    """Factory creating an LLM provider based on configuration."""
    cfg = config or settings
    active_type = (provider_type or cfg.llm_provider).lower()

    if active_type == "mock":
        return MockLLMProvider(**kwargs)
    elif active_type == "openai":
        return OpenAILLMProvider(
            api_key=kwargs.get("api_key") or cfg.openai_api_key,
            model_name=kwargs.get("model_name") or cfg.llm_model_name,
            temperature=kwargs.get("temperature", cfg.llm_temperature),
        )
    elif active_type == "anthropic":
        return AnthropicLLMProvider(
            api_key=kwargs.get("api_key") or cfg.anthropic_api_key,
            model_name=kwargs.get("model_name") or "claude-3-5-sonnet-20241022",
            temperature=kwargs.get("temperature", cfg.llm_temperature),
        )
    elif active_type == "ollama":
        return OllamaLLMProvider(
            base_url=kwargs.get("base_url") or cfg.ollama_base_url,
            model_name=kwargs.get("model_name") or "llama3.1:8b",
            temperature=kwargs.get("temperature", cfg.llm_temperature),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {active_type}")
