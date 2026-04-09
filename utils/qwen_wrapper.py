"""
Qwen LLM Wrapper - 轻量级版本
将 QwenClient 包装为 LangChain compatible LLM
"""

from typing import Any, List, Optional
from langchain.llms.base import LLM


class QwenLLMWrapper(LLM):
    """
    Qwen LLM Wrapper for LangChain (轻量级)

    将原有的 QwenClient 包装为 LangChain compatible LLM
    """
    qwen_client: Any  # QwenClient instance
    temperature: float = 0.1
    enable_search: bool = False  # 默认关闭搜索
    force_search: bool = False
    enable_thinking: bool = False
    search_strategy: str = "max"

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "qwen3-max"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        调用 Qwen LLM

        Args:
            prompt: 输入 prompt
            stop: 停止词列表
            **kwargs: 其他参数，支持:
                - temperature: 温度参数
                - enable_search: 是否启用搜索
                - force_search: 是否强制搜索
                - enable_thinking: 是否启用思考
                - search_strategy: 搜索策略

        Returns:
            LLM 响应
        """
        # 将 prompt 转换为 messages 格式
        messages = [{"role": "user", "content": prompt}]

        # 调用 QwenClient，传递搜索相关参数
        # 优先使用实例属性，但允许 kwargs 覆盖
        try:
            response = self.qwen_client.chat(
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                enable_search=kwargs.get('enable_search', self.enable_search),
                force_search=kwargs.get('force_search', self.force_search),
                enable_thinking=kwargs.get('enable_thinking', self.enable_thinking),
                search_strategy=kwargs.get('search_strategy', self.search_strategy)
            )
            return response

        except Exception as e:
            return f"Error calling Qwen: {str(e)}"
