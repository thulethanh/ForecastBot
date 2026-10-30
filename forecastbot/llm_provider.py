from __future__ import annotations
import time, requests
from dataclasses import dataclass
from typing import Any, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from forecastbot.config import get_settings

@dataclass
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    duration_seconds: float = 0.0

class LLMProvider:
    def __init__(self, provider: str = "gemini", model: str | None = None):
        s = get_settings(); self.provider = provider.lower().strip(); self.settings=s
        if self.provider == "gemini":
            self.model = model or s.gemini_model
            self.client = ChatGoogleGenerativeAI(model=self.model, google_api_key=s.google_api_key, temperature=0)
        elif self.provider in {"gemma", "ollama"}:
            self.provider = "gemma"; self.model = model or s.local_forecast_model; self.client=None
        else: raise ValueError(f"Unsupported provider: {provider}")

    def invoke(self, prompt: str) -> LLMResponse:
        t0=time.perf_counter()
        if self.provider == "gemini":
            r=self.client.invoke(prompt); content=r.content if hasattr(r,"content") else str(r)
            meta=getattr(r,"usage_metadata",{}) or {}; resp=getattr(r,"response_metadata",{}) or {}
            usage=resp.get("usage_metadata",{}) if isinstance(resp,dict) else {}
            inp=meta.get("input_tokens", usage.get("prompt_token_count")); out=meta.get("output_tokens", usage.get("candidates_token_count")); total=meta.get("total_tokens", usage.get("total_token_count"))
        else:
            url=self.settings.ollama_base_url.rstrip("/")+"/api/generate"
            rr=requests.post(url,json={"model":self.model,"prompt":prompt,"stream":False,"options":{"temperature":0}},timeout=self.settings.local_forecast_timeout); rr.raise_for_status(); data=rr.json(); content=data.get("response",""); inp=data.get("prompt_eval_count"); out=data.get("eval_count"); total=(inp+out) if isinstance(inp,int) and isinstance(out,int) else None
        return LLMResponse(content=content,input_tokens=inp,output_tokens=out,total_tokens=total,duration_seconds=time.perf_counter()-t0)
