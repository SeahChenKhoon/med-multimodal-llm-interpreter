from typing import Any, Dict
from openai import OpenAI, AzureOpenAI
import litellm

class LLMClient:
    def __init__(
        self,
        provider: str = None,
        api_key: str = None,        
        model: str  = None,
        messages: list =[],
        extra_params=None
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.messages = messages
        self.extra_params = extra_params
    def completion(self):
        response=""
        kwargs = {
            "model": self.model,
            "messages": self.messages,
            "api_key": self.api_key
        }
        if self.extra_params:
            kwargs.update(self.extra_params)
        response = litellm.completion(**kwargs)
        return _strip_markdown_fences(response['choices'][0]['message']['content'])
    

def _strip_markdown_fences(text: str) -> str:
    lines = text.strip().splitlines()
    cleaned_lines = []

    for line in lines:
        if line.strip().startswith("```"):
            continue  # Skip the fence line
        cleaned_lines.append(line)


    return "\n".join(cleaned_lines)


def build_llm_client(settings_dict: dict, prompt: str) -> LLMClient:
    provider = settings_dict["provider"].lower()

    if provider == "azure":
        return LLMClient(
            provider=provider,
            api_key=settings_dict["azure_openai_api_key"],
            model=settings_dict["llm_model"],
            messages=[{"role": "user", "content": prompt}],
            extra_params={
                "api_base": settings_dict["azure_openai_endpoint"],
                "deployment_id": settings_dict["azure_openai_deployment"]
            }
        )
    else:  # assume "openai"
        return LLMClient(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key=settings_dict["openai_api_key"],
            messages=[{"role": "user", "content": prompt}]
        )

    