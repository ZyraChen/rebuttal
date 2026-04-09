"""
ClaimDecomposer - Claim分解器

将原始claimdecomposed into多个需要验证的子claim
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from typing import Dict, Optional

import uuid
import json
import re
from typing import List

# 从mad目录导入共享组件
from utils.simple_prompt import SimplePromptTemplate as PromptTemplate
from utils.simple_chain import SimpleLLMChain as LLMChain

# 从mad_v2导入模型
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.models import SubClaim


class ClaimDecomposer:
    """
    Claim分解器

    将一个复杂的claimdecomposed into多个子claim，每个子claim代表一个需要验证的断言
    """

    def __init__(self, llm):
        """
        初始化

        Args:
            llm: LangChain compatible LLM (QwenLLMWrapper)
        """
        self.llm = llm

        # Claim分解的Prompt
        self.decompose_template = PromptTemplate(
            input_variables=["claim"],
            template="""You are a fact-checking expert. Your task is to decompose a complex claim into verifiable sub-claims.

**Original Claim:**
{claim}

**Task:**
Analyze this claim and break it down into 2-4 atomic sub-claims that need to be verified independently.

**Decomposition Principles:**
1. **Atomic**: Each sub-claim should be a single, independent assertion
2. **Verifiable**: Each sub-claim can be verified through evidence search
3. **Complete**: Together, all sub-claims should cover the full scope of the original claim
4. **Non-Redundant**: Do not repeat the same fact in multiple sub-claims.
5. **Coverage-Oriented**: Taken together, the sub-claims should fully determine whether the original claim is true or false.
6. **No Inference**: Do NOT add new facts, explanations, or causal reasoning beyond what is explicitly stated or strictly implied by the original claim.

**Sub-Claim Types:**
- **Entity**: Who/what is the main subject? (e.g., "X is a person/organization/thing")
- **Event**: Did something happen? (e.g., "Event X occurred")
- **Time**: When did it happen? (e.g., "Event happened in 2020")
- **Quantity**: How much/many? (e.g., "X has 23 Grand Slam titles")
- **Qualifier**: Scope/degree modifiers (e.g., "most in history" vs "most in Open Era")
- **Relationship**: Connection between entities (e.g., "X is CEO of Y")


**Output Format (JSON only, no other text, Ensure the JSON is complete and parsed correctly.):**
```json
{{
  "sub_claims": [
    {{
      "text": "Serena Williams is a female tennis player",//Maintain the same language as the original claim
      "verification_type": "Entity",
      "rationale": "Core identity of the subject"
    }},
    {{
      "text": "Serena Williams has won the most Grand Slam singles titles",//Maintain the same language as the original claim
      "verification_type": "Quantity",
      "rationale": "Core assertion about the record"
    }},
    {{
      "text": "This record is for all-time history",//Maintain the same language as the original claim
      "verification_type": "Qualifier",
      "rationale": "Critical scope qualifier"
    }}
  ]
}}
```

**IMPORTANT**: Return ONLY valid JSON. Do not include any text before or after the JSON object.
- Focus on the MAIN assertions - don't over-decompose

Now decompose the claim:"""
        )

        self.decompose_chain = LLMChain(llm=self.llm, prompt=self.decompose_template)

    def decompose(self, claim: str) -> List[SubClaim]:
        """
        分解claim为子claims

        Args:
            claim: 原始claim

        Returns:
            List[SubClaim]: 子claim列表
        """
        # 禁用搜索（这是分析任务）
        original_search = self.llm.enable_search
        original_force = self.llm.force_search
        self.llm.enable_search = False
        self.llm.force_search = False

        try:
            # 启用thinking模式以获得更好的分解
            result = self.decompose_chain.invoke(
                inputs={"claim": claim},
                llm_kwargs={"enable_thinking": True}
            )

            text = result.get('text', '')

            print("完整LLM输出:")
            print(repr(text))  # 使用repr可以看到所有转义字符

            # 解析JSON输出
            sub_claims_data = self._parse_decomposition_output(text)

            if not sub_claims_data or 'sub_claims' not in sub_claims_data:
                print("⚠ 无法解析子claim，使用默认分解")
                return self._default_decomposition(claim)

            # 创建SubClaim对象
            sub_claims = []
            for i, sc_data in enumerate(sub_claims_data['sub_claims']):
                if 'text' not in sc_data:
                    continue

                sub_claim = SubClaim(
                    id=f"subclaim_{uuid.uuid4().hex[:8]}",
                    text=sc_data['text'].strip(),
                    parent_claim=claim,
                    verification_type=sc_data.get('verification_type', 'Unknown')
                )
                sub_claims.append(sub_claim)

                print(f"  ✓ 子Claim {i+1}: {sub_claim.text}")
                print(f"    类型: {sub_claim.verification_type}, importance: {sub_claim.importance}")

            return sub_claims

        except Exception as e:
            print(f"⚠ Claimdecomposition failed: {e}")
            return self._default_decomposition(claim)

        finally:
            # 恢复原始配置
            self.llm.enable_search = original_search
            self.llm.force_search = original_force

    def _parse_decomposition_output(self, text: str) -> dict:
        """解析LLM输出的JSON格式 - 改进版"""

        # 策略1: 提取markdown代码块
        code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1).strip()
            parsed = self._try_parse_json(json_str)
            if parsed:
                return parsed

        # 策略2: 直接解析整个文本
        parsed = self._try_parse_json(text.strip())
        if parsed:
            return parsed

        # 策略3: 寻找最外层的 { ... } 对
        json_str = self._extract_outermost_json(text)
        if json_str:
            parsed = self._try_parse_json(json_str)
            if parsed:
                return parsed

        # 策略4: 修复常见error后重试
        json_str = self._fix_common_json_errors(text)
        parsed = self._try_parse_json(json_str)
        if parsed:
            return parsed

        print(f"  ⚠ 所有JSON解析策略failed")
        print(f"  原始输出长度: {len(text)} 字符")
        print(f"  前200字符: {text[:200]}")
        print(f"  后200字符: {text[-200:]}")

        return {}

    def _try_parse_json(self, json_str: str) -> Optional[Dict]:
        """尝试解析JSON，带验证"""
        try:
            # 先做基本的括号完整性检查
            if not self._is_json_complete(json_str):
                return None

            parsed = json.loads(json_str)

            # 验证必须包含sub_claims字段
            if 'sub_claims' in parsed and isinstance(parsed['sub_claims'], list):
                # 进一步验证sub_claims的结构
                if self._validate_subclaims(parsed['sub_claims']):
                    return parsed
        except json.JSONDecodeError as e:
            # 可选：打印详细error信息用于调试
            # print(f"    JSON解析error at pos {e.pos}: {e.msg}")
            pass
        except Exception:
            pass

        return None

    def _is_json_complete(self, json_str: str) -> bool:
        """检查JSON结构是否完整"""
        # 检查花括号配对
        if json_str.count('{') != json_str.count('}'):
            return False

        # 检查方括号配对
        if json_str.count('[') != json_str.count(']'):
            return False

        # 检查双引号配对（排除转义的引号）
        cleaned = json_str.replace('\\"', '')
        if cleaned.count('"') % 2 != 0:
            return False

        return True

    def _validate_subclaims(self, sub_claims: list) -> bool:
        """验证sub_claims数组的结构"""
        if not sub_claims:
            return False

        required_fields = {'text', 'verification_type', 'rationale'}

        for claim in sub_claims:
            if not isinstance(claim, dict):
                return False
            if not required_fields.issubset(claim.keys()):
                return False
            if not all(isinstance(claim[k], str) for k in required_fields):
                return False

        return True

    def _extract_outermost_json(self, text: str) -> Optional[str]:
        """提取最外层的完整JSON对象（支持嵌套）"""
        # 找到Round一个 {
        start = text.find('{')
        if start == -1:
            return None

        # 使用栈来匹配括号
        brace_count = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            char = text[i]

            # Processing转义字符
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            # Processing字符串
            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            # 计数括号
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1

                # 找到匹配的右括号
                if brace_count == 0:
                    return text[start:i + 1]

        return None

    def _fix_common_json_errors(self, text: str) -> str:
        """修复常见的JSONerror（保守策略）"""

        # 1. 提取代码块（如果有）
        code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if code_block_match:
            text = code_block_match.group(1)

        # 2. 移除注释
        text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

        # 3. 移除尾随逗号
        text = re.sub(r',(\s*[}\]])', r'\1', text)

        # 4. 修复字符串内的换行符（这是你的主要问题之一）
        # 将字符串值中的实际换行替换为空格
        def fix_string_breaks(match):
            content = match.group(1)
            # 只替换实际的换行，保留 \n 转义序列
            fixed = content.replace('\n', ' ').replace('\r', '')
            return f'"{fixed}"'

        # 匹配双引号内的内容（不包括已转义的引号）
        text = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', fix_string_breaks, text)

        return text.strip()

    def _default_decomposition(self, claim: str) -> List[SubClaim]:
        """
        默认分解策略（当LLMfailed时）
        简单地将整个claim作为单个子claim
        """
        return [
            SubClaim(
                id=f"subclaim_{uuid.uuid4().hex[:8]}",
                text=claim,
                parent_claim=claim,
                verification_type="Overall"
            )
        ]
