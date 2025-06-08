from typing import Any, Dict, List, Optional, Type, TypeVar
from openai import OpenAI, AzureOpenAI
import litellm
from litellm import acompletion
import instructor
from pydantic import BaseModel


T = TypeVar('T', bound=BaseModel)


class LLMClient:
    def __init__(
        self,
        provider: str = None,
        api_key: str = None,
        model: str = None,
        messages: list = [],
        extra_params=None
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.messages = messages
        self.extra_params = extra_params
        self.instructor_client = instructor.from_litellm(litellm.acompletion)

    def completion(self):
        response = ""
        kwargs = {
            "model": self.model,
            "messages": self.messages,
            "api_key": self.api_key
        }
        if self.extra_params:
            kwargs.update(self.extra_params)
        response = litellm.completion(**kwargs)
        return _strip_markdown_fences(response['choices'][0]['message']['content'])

    def structured_completion(
        self, response_model: Type[T], temperature: float = 0.3
    ) -> T:
        """New method for structured responses using instructor"""
        kwargs = {
            "model": self.model,
            "messages": self.messages,
            "api_key": self.api_key,
            "response_model": response_model,
            "temperature": temperature
        }
        if self.extra_params:
            kwargs.update(self.extra_params)
        
        return self.instructor_client.chat.completions.create(**kwargs)
    
    async def async_structured_completion(
        self, response_model: Type[T], temperature: float = 0.3
    ) -> T:
        """Async method for structured responses using instructor"""
        kwargs = {
            "model": self.model,
            "messages": self.messages,
            "api_key": self.api_key,
            "response_model": response_model,
            "temperature": temperature
        }
        if self.extra_params:
            kwargs.update(self.extra_params)
        
        return await self.instructor_client.chat.completions.create(**kwargs)
    
    @staticmethod
    def run_prompt(settings_dict: dict, prompt_template: str, prompt_context: dict) -> str:
        prompt = prompt_template.format(**prompt_context)
        llm_client = build_llm_client(settings_dict, prompt)
        response = llm_client.completion()
        return response
        
   
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

