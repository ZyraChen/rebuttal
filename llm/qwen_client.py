"""
LLM API Client Wrapper - Support for Alibaba Cloud Qwen
"""

from openai import OpenAI
from typing import List, Dict, Optional
import json
import os


class QwenClient:
    """LLM API Wrapper - Using Alibaba Cloud Qwen"""

    def __init__(self, api_key: str = "sk-350e5068abb745919baa79e2673ce763", model: str = "qwen3-max"):
        # If api_key is not provided, get it from environment variable
        api_key = api_key

        # Alibaba Cloud Qwen uses OpenAI-compatible interface
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model
        print(f"[LLM] Using model: {model}, API Key: {api_key[:8]}...")

    def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 50000,
        enable_search: bool = False,
        force_search: bool = False,
        enable_thinking: bool = False,
        search_strategy: str = "turbo"
    ) -> str:
        """
        Standard chat interface

        Args:
            messages: List of conversation messages
            system: System prompt
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            enable_search: Whether to enable search functionality (default False)
            force_search: Whether to force search (default False, only effective when enable_search=True)
            enable_thinking: Whether to enable thinking mode (default True)
            search_strategy: Search strategy "max"
        """
        # If system prompt exists, add it to the beginning of messages
        if system:
            messages = [{"role": "system", "content": system}] + messages

        try:
            # Build extra_body based on parameters
            extra_body = {}

            if enable_thinking:
                extra_body["enable_thinking"] = True

            if enable_search:
                extra_body["enable_search"] = True
                extra_body["search_options"] = {
                    "forced_search": force_search,
                    "search_strategy": search_strategy
                }

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body=extra_body if extra_body else None,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Qwen API call failed: {e}")
            return ""

    def chat_with_json(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.1
    ) -> dict:
        """
        Request JSON format response
        """
        if system:
            system += "\n\nYou must return valid JSON format, do not include any other text."
        else:
            system = "You must return valid JSON format, do not include any other text."

        response_text = self.chat(messages, system, temperature)

        # Try to parse JSON
        try:
            # Remove possible markdown code blocks
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            return json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Original response: {response_text[:500]}")
            return {}

