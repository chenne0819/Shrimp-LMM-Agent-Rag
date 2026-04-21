from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

RuntimeMode = Literal["workspace", "service"]

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
AGENT_ASSETS_DIR: Final[Path] = PROJECT_ROOT / ".deepagents"

MODEL_ENV_VAR: Final[str] = "DEEPAGENT_MODEL"
MODEL_PROVIDER_ENV_VAR: Final[str] = "DEEPAGENT_MODEL_PROVIDER"
TEMPERATURE_ENV_VAR: Final[str] = "DEEPAGENT_TEMPERATURE"
MAX_TOKENS_ENV_VAR: Final[str] = "DEEPAGENT_MAX_TOKENS"
BACKEND_MODE_ENV_VAR: Final[str] = "DEEPAGENT_BACKEND_MODE"


def _read_optional_float(env_name: str) -> float | None:
    raw_value = os.getenv(env_name)
    if raw_value is None or raw_value == "":
        return None

    try:
        return float(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{env_name} must be a number, got: {raw_value}") from exc


def _read_optional_int(env_name: str) -> int | None:
    raw_value = os.getenv(env_name)
    if raw_value is None or raw_value == "":
        return None

    try:
        return int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{env_name} must be an integer, got: {raw_value}") from exc


def _read_runtime_mode(value: str | None) -> RuntimeMode | None:
    if value is None or value == "":
        return None

    if value not in {"workspace", "service"}:
        raise RuntimeError(
            f"{BACKEND_MODE_ENV_VAR} must be 'workspace' or 'service', got: {value}"
        )

    return value


@dataclass(frozen=True)
class DeepAgentSettings:
    model: str
    model_provider: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    runtime_mode: RuntimeMode = "workspace"

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        model_provider: str | None = None,
        runtime_mode: RuntimeMode | None = None,
    ) -> "DeepAgentSettings":
        selected_model = model or os.getenv(MODEL_ENV_VAR)
        if not selected_model:
            raise RuntimeError(
                "Deep Agents 需要支援 tool calling 的 LangChain chat model。"
                f"請先設定 {MODEL_ENV_VAR}，例如 openai:gpt-5.4 或 google_genai:gemini-3.1-pro-preview。"
            )

        return cls(
            model=selected_model,
            model_provider=model_provider or os.getenv(MODEL_PROVIDER_ENV_VAR),
            temperature=_read_optional_float(TEMPERATURE_ENV_VAR),
            max_tokens=_read_optional_int(MAX_TOKENS_ENV_VAR),
            runtime_mode=runtime_mode
            or _read_runtime_mode(os.getenv(BACKEND_MODE_ENV_VAR))
            or "workspace",
        )
