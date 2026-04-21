import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def import_module_fresh(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class DeepAgentArchitectureTests(unittest.TestCase):
    def tearDown(self):
        for module_name in [
            "agent",
            "deepagent_factory",
            "deepagent_settings",
            "deepagent_tools",
            "deepagents",
            "deepagents.backends",
            "deepagents.backends.utils",
            "langchain",
            "langchain.chat_models",
        ]:
            sys.modules.pop(module_name, None)

    def _patch_deepagent_dependencies(self):
        calls = {}

        class FakeFilesystemBackend:
            def __init__(self, root_dir, virtual_mode=False):
                self.root_dir = root_dir
                self.virtual_mode = virtual_mode

        def fake_init_chat_model(*, model, **kwargs):
            calls["init_chat_model"] = {"model": model, "kwargs": kwargs}
            return "fake_chat_model"

        def fake_create_deep_agent(**kwargs):
            calls["create_deep_agent"] = kwargs
            return types.SimpleNamespace(invoke=lambda payload, config=None: {"messages": ["ok"]})

        def fake_create_file_data(content: str):
            calls.setdefault("create_file_data", []).append(content)
            return {"content": content}

        fake_deepagents = types.ModuleType("deepagents")
        fake_deepagents.create_deep_agent = fake_create_deep_agent

        fake_backends = types.ModuleType("deepagents.backends")
        fake_backends.FilesystemBackend = FakeFilesystemBackend

        fake_backends_utils = types.ModuleType("deepagents.backends.utils")
        fake_backends_utils.create_file_data = fake_create_file_data

        fake_langchain = types.ModuleType("langchain")
        fake_langchain_chat_models = types.ModuleType("langchain.chat_models")
        fake_langchain_chat_models.init_chat_model = fake_init_chat_model

        return calls, mock.patch.dict(
            sys.modules,
            {
                "deepagents": fake_deepagents,
                "deepagents.backends": fake_backends,
                "deepagents.backends.utils": fake_backends_utils,
                "langchain": fake_langchain,
                "langchain.chat_models": fake_langchain_chat_models,
            },
            clear=False,
        )

    def test_settings_from_env_reads_runtime_configuration(self):
        settings_module = import_module_fresh("deepagent_settings")

        with mock.patch.dict(
            os.environ,
            {
                "DEEPAGENT_MODEL": "microsoft/Phi-3-mini-4k-instruct",
                "DEEPAGENT_MODEL_PROVIDER": "huggingface",
                "DEEPAGENT_TEMPERATURE": "0.2",
                "DEEPAGENT_MAX_TOKENS": "512",
                "DEEPAGENT_BACKEND_MODE": "service",
            },
            clear=True,
        ):
            settings = settings_module.DeepAgentSettings.from_env()

        self.assertEqual(settings.model, "microsoft/Phi-3-mini-4k-instruct")
        self.assertEqual(settings.model_provider, "huggingface")
        self.assertEqual(settings.temperature, 0.2)
        self.assertEqual(settings.max_tokens, 512)
        self.assertEqual(settings.runtime_mode, "service")

    def test_settings_from_env_rejects_invalid_runtime_mode(self):
        settings_module = import_module_fresh("deepagent_settings")

        with mock.patch.dict(
            os.environ,
            {
                "DEEPAGENT_MODEL": "openai:gpt-4o-mini",
                "DEEPAGENT_BACKEND_MODE": "invalid-mode",
            },
            clear=True,
        ):
            with self.assertRaises(RuntimeError) as exc_info:
                settings_module.DeepAgentSettings.from_env()

        self.assertIn("DEEPAGENT_BACKEND_MODE", str(exc_info.exception))

    def test_workspace_agent_uses_filesystem_backend(self):
        calls, patcher = self._patch_deepagent_dependencies()
        with patcher:
            settings_module = import_module_fresh("deepagent_settings")
            factory_module = import_module_fresh("deepagent_factory")
            settings = settings_module.DeepAgentSettings(
                model="openai:gpt-4o-mini",
                runtime_mode="workspace",
            )

            built_agent = factory_module.build_agent(settings)

        self.assertIsNotNone(built_agent)
        self.assertEqual(calls["init_chat_model"]["model"], "openai:gpt-4o-mini")
        self.assertEqual(calls["create_deep_agent"]["model"], "fake_chat_model")
        self.assertEqual(calls["create_deep_agent"]["memory"], ["/.deepagents/AGENTS.md"])
        self.assertEqual(calls["create_deep_agent"]["skills"], ["/.deepagents/skills/"])
        self.assertIn("不要另外手刻新的 agent 架構", calls["create_deep_agent"]["system_prompt"])
        self.assertEqual(
            calls["create_deep_agent"]["backend"].root_dir,
            str(PROJECT_ROOT),
        )
        self.assertTrue(calls["create_deep_agent"]["backend"].virtual_mode)

    def test_service_agent_uses_seeded_state_files_instead_of_workspace_backend(self):
        calls, patcher = self._patch_deepagent_dependencies()
        with patcher:
            settings_module = import_module_fresh("deepagent_settings")
            factory_module = import_module_fresh("deepagent_factory")
            factory_module.get_service_seed_files.cache_clear()
            settings = settings_module.DeepAgentSettings(
                model="openai:gpt-4o-mini",
                runtime_mode="service",
            )

            payload = factory_module.build_agent_input("help", settings)
            factory_module.build_agent(settings)

        self.assertIn("files", payload)
        self.assertIn("/.deepagents/AGENTS.md", payload["files"])
        self.assertIn("/.deepagents/skills/calculator/SKILL.md", payload["files"])
        self.assertNotIn("backend", calls["create_deep_agent"])
        self.assertGreaterEqual(len(calls["create_file_data"]), 2)

    def test_invoke_agent_passes_thread_config(self):
        settings_module = import_module_fresh("deepagent_settings")
        factory_module = import_module_fresh("deepagent_factory")
        settings = settings_module.DeepAgentSettings(
            model="openai:gpt-4o-mini",
            runtime_mode="service",
        )

        fake_agent = mock.Mock()
        fake_agent.invoke.return_value = {"messages": [types.SimpleNamespace(content="done")]}

        with mock.patch.object(factory_module, "get_agent", return_value=fake_agent), mock.patch.object(
            factory_module,
            "build_agent_input",
            return_value={"messages": [{"role": "user", "content": "help"}]},
        ):
            result = factory_module.invoke_agent("help", settings=settings, thread_id="thread-1")

        self.assertEqual(result, "done")
        fake_agent.invoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "help"}]},
            config={"configurable": {"thread_id": "thread-1"}},
        )

    def test_deepagents_files_follow_expected_names(self):
        agents_path = PROJECT_ROOT / ".deepagents" / "AGENTS.md"
        skill_path = PROJECT_ROOT / ".deepagents" / "skills" / "calculator" / "SKILL.md"

        self.assertTrue(agents_path.exists())
        self.assertTrue(skill_path.exists())

        skill_text = skill_path.read_text(encoding="utf-8")
        self.assertTrue(skill_text.startswith("---"))
        self.assertIn("name: calculator", skill_text)
        self.assertIn("description:", skill_text)

    def test_calculate_math_supports_basic_arithmetic(self):
        tools_module = import_module_fresh("deepagent_tools")
        self.assertEqual(tools_module.calculate_math("12 * (3 + 4)"), "84")

    def test_calculate_math_rejects_non_arithmetic_input(self):
        tools_module = import_module_fresh("deepagent_tools")
        result = tools_module.calculate_math("__import__('os').system('dir')")
        self.assertIn("計算失敗", result)


if __name__ == "__main__":
    unittest.main()
