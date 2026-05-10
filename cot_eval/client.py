from __future__ import annotations

import time
from typing import Any

from cot_eval.types import CompletionResult


class OpenAIChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        reasoning_effort: str | None,
        max_tokens: int,
        timeout: float,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package to run model-backed evaluations.") from exc

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._max_tokens = max_tokens

    def complete(self, prompt: str) -> CompletionResult:
        extra_body = {}
        if self._reasoning_effort:
            extra_body["reasoning_effort"] = self._reasoning_effort

        start = time.perf_counter()
        retried = False
        try:
            response = self._create(prompt, extra_body=extra_body or None)
        except Exception:
            if not extra_body:
                raise
            retried = True
            response = self._create(prompt, extra_body=None)
        latency_s = time.perf_counter() - start

        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        discarded_reasoning = hasattr(message, "reasoning_content") or hasattr(message, "reasoning")
        usage = getattr(response, "usage", None)
        return CompletionResult(
            content=content,
            latency_s=latency_s,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            retried_without_extra_body=retried,
            discarded_reasoning=discarded_reasoning,
        )

    def _create(self, prompt: str, extra_body: dict[str, Any] | None) -> Any:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self._max_tokens,
        }
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        return self._client.chat.completions.create(**kwargs)
