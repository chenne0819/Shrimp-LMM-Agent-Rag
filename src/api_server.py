from __future__ import annotations

from functools import lru_cache
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from deepagent_factory import describe_runtime_assets, invoke_agent
from deepagent_settings import DeepAgentSettings

app = FastAPI(title="Shrimp LMM Agent Service")


class GemmaChatRequest(BaseModel):
    prompt: str
    file_path: Optional[str] = None
    enable_thinking: bool = False


class DeepAgentRequest(BaseModel):
    prompt: str
    thread_id: str = Field(default="default")
    model: Optional[str] = None
    model_provider: Optional[str] = None


@lru_cache(maxsize=1)
def get_gemma_runtime():
    import torch

    from lmm import model, prepare_multimodal_messages, processor

    return {
        "torch": torch,
        "model": model,
        "processor": processor,
        "prepare_multimodal_messages": prepare_multimodal_messages,
    }


@app.get("/health")
async def healthcheck():
    return {"status": "ok"}


@app.get("/deep-agent/health")
async def deep_agent_healthcheck():
    try:
        return {
            "status": "ok",
            "assets": describe_runtime_assets(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat")
async def gemma_chat_endpoint(request: GemmaChatRequest):
    try:
        gemma_runtime = get_gemma_runtime()
        messages = gemma_runtime["prepare_multimodal_messages"](
            prompt=request.prompt,
            file_path=request.file_path,
        )

        processor = gemma_runtime["processor"]
        model = gemma_runtime["model"]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=request.enable_thinking,
            processor_kwargs={
                "video_kwargs": {"num_frames": 8},
            },
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        with gemma_runtime["torch"].no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                use_cache=True,
            )

        response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        return {"response": response}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/deep-agent/chat")
async def deep_agent_chat_endpoint(request: DeepAgentRequest):
    try:
        settings = DeepAgentSettings.from_env(
            model=request.model,
            model_provider=request.model_provider,
            runtime_mode="service",
        )
        response = invoke_agent(
            request.prompt,
            settings=settings,
            thread_id=request.thread_id,
        )
        return {
            "response": response,
            "thread_id": request.thread_id,
            "model": settings.model,
            "runtime_mode": settings.runtime_mode,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
