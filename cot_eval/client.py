from __future__ import annotations

import time
import sys
from typing import Any

from cot_eval.types import CompletionResult


class EndpointUnavailableError(RuntimeError):
    """Raised when the OpenAI-compatible endpoint cannot be reached."""


class OpenAIChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        reasoning_effort: str | None,
        max_tokens: int,
        timeout: float,
        system_prompt: str | None = None,
    ) -> None:
        try:
            from openai import APIConnectionError, APITimeoutError, BadRequestError, OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package to run model-backed evaluations.") from exc

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._connection_errors = (APIConnectionError, APITimeoutError)
        self._bad_request_error = BadRequestError

    def check_connection(self, timeout: float = 5.0) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Install the 'httpx' package to run endpoint preflight checks.") from exc

        models_url = f"{self._base_url.rstrip('/')}/models"
        headers = {}
        if self._api_key and self._api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            response = httpx.get(models_url, headers=headers, timeout=timeout)
            response.raise_for_status()
            models_payload = response.json()
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.NetworkError) as exc:
            raise EndpointUnavailableError(
                f"Could not connect to the OpenAI-compatible endpoint at {self._base_url}. "
                "Start vLLM first, or pass --base-url for the server you want to use.\n\n"
                "Suggested local 4-GPU server command:\n"
                "CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve openai/gpt-oss-20b "
                "--host 0.0.0.0 --port 8000 --tensor-parallel-size 4 "
                "--max-model-len 8192 --gpu-memory-utilization 0.92 "
                "--disable-custom-all-reduce"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise EndpointUnavailableError(
                f"Endpoint {models_url} responded with HTTP {exc.response.status_code}. "
                "Confirm --base-url points to an OpenAI-compatible /v1 endpoint."
            ) from exc
        except ValueError as exc:
            raise EndpointUnavailableError(f"Endpoint {models_url} did not return valid JSON.") from exc

        available_models = [model.get("id", "") for model in models_payload.get("data", []) if isinstance(model, dict)]
        if available_models and self._model not in available_models:
            print(
                f"Warning: model {self._model!r} was not listed by {self._base_url}/models. "
                f"Available models: {', '.join(available_models)}",
                file=sys.stderr,
            )

    def complete(self, prompt: str) -> CompletionResult:
        extra_body = {}
        if self._reasoning_effort:
            extra_body["reasoning_effort"] = self._reasoning_effort

        start = time.perf_counter()
        retried = False
        try:
            response = self._create(prompt, extra_body=extra_body or None)
        except Exception as exc:
            self._raise_endpoint_error_if_needed(exc)
            if not extra_body or not self._should_retry_without_extra_body(exc):
                raise
            retried = True
            try:
                response = self._create(prompt, extra_body=None)
            except Exception as retry_exc:
                self._raise_endpoint_error_if_needed(retry_exc)
                raise
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
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self._max_tokens,
        }
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        return self._client.chat.completions.create(**kwargs)

    def _raise_endpoint_error_if_needed(self, exc: Exception) -> None:
        if isinstance(exc, self._connection_errors):
            raise EndpointUnavailableError(
                f"Lost connection to the OpenAI-compatible endpoint at {self._base_url}. "
                "Confirm the vLLM server is still running and reachable."
            ) from exc

    def _should_retry_without_extra_body(self, exc: Exception) -> bool:
        if not isinstance(exc, self._bad_request_error):
            return False
        message = str(exc).lower()
        return "extra_body" in message or "reasoning_effort" in message or "unknown" in message
