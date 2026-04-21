from __future__ import annotations

from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Final, Iterable

from deepagent_settings import AGENT_ASSETS_DIR, DeepAgentSettings, PROJECT_ROOT
from deepagent_tools import calculate_math

MEMORY_PATHS: Final[list[str]] = ["/.deepagents/AGENTS.md"]
SKILL_PATHS: Final[list[str]] = ["/.deepagents/skills/"]
SYSTEM_PROMPT: Final[str] = (
    "你是 Shrimp-LMM-Agent-Rag 專案的 Deep Agent。"
    "請遵守 /.deepagents/AGENTS.md 的專案規則，"
    "優先使用 Deep Agents 內建的 planning、filesystem、memory 與 skills 能力，"
    "不要另外手刻新的 agent 架構。"
)
def iter_asset_files() -> Iterable[tuple[str, Path]]:
    yield MEMORY_PATHS[0], AGENT_ASSETS_DIR / "AGENTS.md"

    skills_root = AGENT_ASSETS_DIR / "skills"
    for disk_path in sorted(path for path in skills_root.rglob("*") if path.is_file()):
        relative_path = disk_path.relative_to(PROJECT_ROOT).as_posix()
        yield f"/{relative_path}", disk_path


def validate_agent_assets() -> None:
    for virtual_path, disk_path in iter_asset_files():
        if not disk_path.is_file():
            raise FileNotFoundError(
                f"Deep Agents asset file not found for {virtual_path}: {disk_path}"
            )


def build_chat_model(settings: DeepAgentSettings):
    from langchain.chat_models import init_chat_model

    init_kwargs = {}
    if settings.model_provider:
        init_kwargs["model_provider"] = settings.model_provider
    if settings.temperature is not None:
        init_kwargs["temperature"] = settings.temperature
    if settings.max_tokens is not None:
        init_kwargs["max_tokens"] = settings.max_tokens

    return init_chat_model(model=settings.model, **init_kwargs)


def build_seed_files() -> dict[str, object]:
    from deepagents.backends.utils import create_file_data

    validate_agent_assets()
    seeded_files: dict[str, object] = {}
    for virtual_path, disk_path in iter_asset_files():
        seeded_files[virtual_path] = create_file_data(disk_path.read_text(encoding="utf-8"))
    return seeded_files


def build_agent(settings: DeepAgentSettings):
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    validate_agent_assets()

    create_kwargs = {
        "model": build_chat_model(settings),
        "tools": [calculate_math],
        "memory": MEMORY_PATHS,
        "skills": SKILL_PATHS,
        "system_prompt": SYSTEM_PROMPT,
    }

    if settings.runtime_mode == "workspace":
        create_kwargs["backend"] = FilesystemBackend(
            root_dir=str(PROJECT_ROOT),
            virtual_mode=True,
        )

    return create_deep_agent(**create_kwargs)


@lru_cache(maxsize=8)
def get_agent(settings: DeepAgentSettings):
    return build_agent(settings)


@lru_cache(maxsize=1)
def get_service_seed_files() -> dict[str, object]:
    return build_seed_files()


def build_agent_input(prompt: str, settings: DeepAgentSettings) -> dict[str, object]:
    payload: dict[str, object] = {
        "messages": [{"role": "user", "content": prompt}],
    }

    if settings.runtime_mode == "service":
        payload["files"] = dict(get_service_seed_files())

    return payload


def build_agent_config(thread_id: str | None = None) -> dict[str, object] | None:
    if not thread_id:
        return None

    return {"configurable": {"thread_id": thread_id}}


def _extract_final_message_content(result: dict[str, object]) -> str:
    messages = result.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RuntimeError("Agent result does not contain any messages.")

    final_message = messages[-1]
    content = getattr(final_message, "content", final_message)
    if isinstance(content, str):
        return content

    return str(content)


def invoke_agent(
    prompt: str,
    *,
    settings: DeepAgentSettings,
    thread_id: str | None = None,
) -> str:
    payload = build_agent_input(prompt, settings)
    config = build_agent_config(thread_id)
    agent = get_agent(settings)

    if config is None:
        result = agent.invoke(payload)
    else:
        result = agent.invoke(payload, config=config)

    return _extract_final_message_content(result)


def describe_runtime_assets() -> dict[str, object]:
    validate_agent_assets()
    return {
        "project_root": str(PROJECT_ROOT),
        "assets_root": str(AGENT_ASSETS_DIR),
        "memory": list(MEMORY_PATHS),
        "skills": list(SKILL_PATHS),
        "seed_files": [str(PurePosixPath(path)) for path, _ in iter_asset_files()],
    }
