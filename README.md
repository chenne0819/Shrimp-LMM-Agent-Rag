# Shrimp-LMM-Agent-Rag

這個專案現在拆成三條清楚的能力線：

- `src/chat_model.py`：本地 Gemma 多模態 CLI 聊天。
- `src/api_server.py`：同時提供 Gemma API 與 Deep Agent API。
- `src/deepagent_*.py`：Deep Agents 官方架構的核心設定、工具與 factory。

## 專案結構

```text
src/
  agent.py                # Deep Agent CLI 入口
  api_server.py           # FastAPI 入口
  chat_model.py           # Gemma 多模態 CLI
  lmm.py                  # Gemma 模型載入
  deepagent_settings.py   # Deep Agent 設定與環境變數
  deepagent_tools.py      # 領域工具
  deepagent_factory.py    # create_deep_agent factory / invoke 封裝
.deepagents/
  AGENTS.md               # 永遠載入的專案規則
  skills/
    calculator/SKILL.md   # 範例 skill
```

## 1. 本地 Gemma 環境

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
pip install torchcodec==0.10
pip install -U transformers librosa accelerate
pip install -U bitsandbytes
```

## 2. Deep Agents 官方接法

目前 Deep Agent 是依照 LangChain Deep Agents 官方文件整理成：

- 使用 `create_deep_agent(...)`
- 使用 `langchain.chat_models.init_chat_model(...)`
- 使用 `.deepagents/AGENTS.md` 與 `.deepagents/skills/`
- `workspace` 模式下用 `FilesystemBackend(root_dir=..., virtual_mode=True)`
- `service` 模式下不直接暴露本機檔案系統，而是把 `.deepagents` 資產 seed 到 request payload 的 `files`

這樣的分工是：

- `deepagents` 負責 planning、filesystem tools、skills、memory hooks、subagent orchestration
- 專案自己只負責 domain rules、skills、tools、API 與資料來源

## 3. 安裝 Deep Agent 依賴

先安裝通用套件：

```bash
pip install -U deepagents langchain
```

再依照你要用的 provider 安裝對應整合：

```bash
# OpenAI
pip install -U "langchain[openai]"

# Google Gemini
pip install -U "langchain[google-genai]"

# Anthropic
pip install -U "langchain[anthropic]"

# Hugging Face Inference
pip install -U "langchain[huggingface]"
```

## 4. Deep Agent 環境變數

至少要設定一個支援 tool calling 的 LangChain chat model：

```powershell
$env:DEEPAGENT_MODEL="openai:gpt-5.4"
```

或：

```powershell
$env:DEEPAGENT_MODEL="google_genai:gemini-3.1-pro-preview"
```

Hugging Face provider 範例：

```powershell
$env:DEEPAGENT_MODEL="microsoft/Phi-3-mini-4k-instruct"
$env:DEEPAGENT_MODEL_PROVIDER="huggingface"
```

可選參數：

```powershell
$env:DEEPAGENT_TEMPERATURE="0.2"
$env:DEEPAGENT_MAX_TOKENS="1024"
```

同時要設定對應 provider 的 API key，例如：

```powershell
$env:OPENAI_API_KEY="..."
```

## 5. 執行方式

Deep Agent CLI：

```bash
python src/agent.py
```

FastAPI：

```bash
python src/api_server.py
```

可用端點：

- `POST /chat`：Gemma 多模態聊天
- `POST /deep-agent/chat`：Deep Agent 對話
- `GET /deep-agent/health`：檢查 Deep Agent 資產是否完整

`/deep-agent/chat` request 範例：

```json
{
  "prompt": "幫我算 123 * 456",
  "thread_id": "demo-thread"
}
```

## 6. 為什麼不再把本地 Gemma 直接包成 Deep Agent 底層

依照官方文件，Deep Agents 需要的是「支援 tool calling 的 LangChain chat model」。

所以：

- 本地 `transformers` text-generation pipeline 不是官方推薦的 Deep Agents 底層接法
- Gemma 多模態流程保留在 `chat_model.py` / `lmm.py`
- Deep Agent 則改成標準的 `create_deep_agent(...)` 架構

## 7. 測試

目前有一組不依賴外部套件與 API 的單元測試，會檢查：

- `DeepAgentSettings` 讀取環境變數是否正確
- `workspace` 模式是否使用 `FilesystemBackend(..., virtual_mode=True)`
- `service` 模式是否把 `.deepagents` 資產 seed 到 `files`
- `init_chat_model` 是否有被用來建立 Deep Agents 模型
- `calculator` skill 對應的工具是否仍可用

執行：

```bash
python -m unittest tests.test_agent_config
```
