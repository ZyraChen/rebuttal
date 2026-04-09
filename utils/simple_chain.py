"""
简单的Chain实现，替代langchain.chains.LLMChain
用于兼容性
"""

class SimpleLLMChain:
    """简单的LLM链，兼容langchain LLMChain接口"""

    def __init__(self, llm, prompt, output_parser=None):
        """
        初始化

        Args:
            llm: LLM实例 (QwenLLMWrapper)
            prompt: PromptTemplate实例
            output_parser: 输出解析器（可选）
        """
        self.llm = llm
        self.prompt = prompt
        self.output_parser = output_parser

    def invoke(self, inputs: dict, llm_kwargs: dict = None) -> dict:
        """
        执行链

        Args:
            inputs: 输入字典，对应prompt的input_variables
            llm_kwargs: 传递给LLM的额外参数（如enable_thinking, temperature等）

        Returns:
            字典，包含'text'字段（如果有output_parser则为解析后的结果）
        """
        # 格式化prompt
        formatted_prompt = self.prompt.format(**inputs)

        # 调用LLM，如果有额外参数则传递
        if llm_kwargs:
            response = self.llm.invoke(formatted_prompt, **llm_kwargs)
        else:
            response = self.llm.invoke(formatted_prompt)

        # 获取文本内容
        if hasattr(response, 'content'):
            text = response.content
        elif isinstance(response, dict):
            text = response.get('content', str(response))
        else:
            text = str(response)

        # 如果有输出解析器，使用它解析
        if self.output_parser:
            parsed = self.output_parser.parse(text)
            return {'text': parsed}

        return {'text': text}
