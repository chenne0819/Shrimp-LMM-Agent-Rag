from __future__ import annotations

from deepagent_factory import invoke_agent
from deepagent_settings import DeepAgentSettings, MODEL_ENV_VAR


def run_agent(prompt: str, *, settings: DeepAgentSettings) -> str | None:
    print(f"\n使用者: {prompt}")
    print("Agent 思考中...")

    try:
        final_message = invoke_agent(prompt, settings=settings)
        print(f"\nAgent 回覆:\n{final_message}")
        return final_message
    except Exception as exc:
        print(f"\nAgent 執行失敗: {exc}")
        print(
            "提示：請先安裝 Deep Agents 與 LangChain provider 套件，"
            "並設定支援 tool calling 的模型與對應 API 金鑰。"
        )
        return None


def main() -> None:
    try:
        settings = DeepAgentSettings.from_env(runtime_mode="workspace")
    except Exception as exc:
        print(f"Deep Agent 初始化失敗: {exc}")
        print(f"請先設定環境變數 {MODEL_ENV_VAR}。")
        return

    print("\nDeep Agent 已啟動，輸入 'exit' 或 'quit' 可離開。")
    print(f"目前設定模型: {settings.model}")
    print("你可以直接問一般問題，或要求它做數學計算，例如 123 * 456。")
    print("-" * 50)

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        if user_input:
            run_agent(user_input, settings=settings)


if __name__ == "__main__":
    main()
