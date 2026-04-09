"""
简单的Prompt Template实现，替代langchain
"""

class SimplePromptTemplate:
    """简单的Prompt模板"""

    def __init__(self, input_variables, template):
        """
        初始化

        Args:
            input_variables: 输入变量列表
            template: 模板字符串
        """
        self.input_variables = input_variables
        self.template = template

    def format(self, **kwargs):
        """格式化模板"""
        return self.template.format(**kwargs)
