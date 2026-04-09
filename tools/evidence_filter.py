"""

证据过滤器



功能：

1. 过滤无关证据（如登录页面、error页面等）

2. 过滤重复证据

3. 使用 LLM 判断证据质量和相关性

"""

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

from typing import List, Dict

from utils.simple_prompt import SimplePromptTemplate as PromptTemplate

from utils.simple_chain import SimpleLLMChain as LLMChain


class EvidenceFilter:
    """Evidence Filter - Use LLM to filter irrelevant and duplicate evidence"""

    def __init__(self, llm):

        """

        初始化



        Args:

            llm: LangChain compatible LLM (QwenLLMWrapper)

        """

        self.llm = llm

        # 证据相关性判断 Prompt

        self.relevance_template = PromptTemplate(

            input_variables=["claim", "evidence_batch"],

            template="""You are a fact-checking expert. Your task is to filter out irrelevant and low-quality evidence.
**Claim to Verify:**
{claim}

**Evidence Batch:**
{evidence_batch}

**Task:**
For each evidence item, determine if it should be KEPT or FILTERED OUT.

**Filter Out if:**
1. **Login/Access Pages**: Content asking to login, sign in, or access restrictions (e.g., "Please login to view", "Sign in to Instagram")
2. **Error Pages**: 404 errors, page not found, server errors
3. **Navigation/UI Only**: Only navigation menus, footers, headers without actual content
4. **Completely Irrelevant**: Content has nothing to do with the claim
5. **Empty/Meaningless**: Empty content, only symbols, or gibberish
6. **Cookie/Privacy Notices**: Only cookie notices or privacy policy without actual content

**Keep if:**
1. Contains factual information related to the claim
2. Contains quotes, data, or evidence that could support or refute the claim
3. Contains background information about entities mentioned in the claim

**Output Format (JSON only):**
```json
{{
  "filtered_evidence": [
    {{
      "evidence_id": "evidence_xxx",
      "keep": true,
      "reason": "Contains relevant data about X"
    }},
    {{
      "evidence_id": "evidence_yyy",
      "keep": false,
      "reason": "Login page requesting authentication"
    }}
  ]
}}

```
**IMPORTANT**: Return ONLY valid JSON. No other text.
Now analyze the evidence:"""

        )

        self.relevance_chain = LLMChain(llm=self.llm, prompt=self.relevance_template)

    def filter_evidence(

            self,

            claim: str,

            evidence_pool: List[Dict],

            batch_size: int = 10

    ) -> List[Dict]:

        """

        过滤证据池



        Args:

            claim: 原始 claim

            evidence_pool: 证据池列表

            batch_size: 每次Processing的证据数量



        Returns:

            过滤后的证据列表

        """

        if not evidence_pool:
            return []

        print(f"\n[证据过滤]")

        print(f"原始证据数量: {len(evidence_pool)}")

        # 先进行基于规则的快速过滤

        evidence_pool = self._rule_based_filter(evidence_pool)

        print(f"规则过滤后: {len(evidence_pool)}")

        # 然后进行去重

        evidence_pool = self._deduplicate_evidence(evidence_pool)

        print(f"去重后: {len(evidence_pool)}")

        # 最后使用 LLM 进行相关性过滤

        filtered_evidence = []

        # 分批Processing

        for i in range(0, len(evidence_pool), batch_size):

            batch = evidence_pool[i:i + batch_size]

            # 构建批次输入

            batch_text = self._format_evidence_batch(batch)

            try:

                # 调用 LLM 判断

                result = self.relevance_chain.invoke(

                    inputs={"claim": claim, "evidence_batch": batch_text},

                    llm_kwargs={"enable_thinking": False}

                )

                text = result.get('text', '')

                decisions = self._parse_filter_output(text)

                # 根据决策保留证据

                evidence_map = {e["id"]: e for e in batch}

                for decision in decisions:

                    eid = decision.get("evidence_id")

                    keep = decision.get("keep", True)

                    reason = decision.get("reason", "")

                    if keep and eid in evidence_map:

                        filtered_evidence.append(evidence_map[eid])

                    elif not keep:

                        print(f"  ✗ 过滤 {eid}: {reason}")



            except Exception as e:

                print(f"⚠ LLM 过滤failed（批次 {i // batch_size + 1}）: {e}")

                # 出错时保留所有证据

                filtered_evidence.extend(batch)

        print(f"LLM 过滤后: {len(filtered_evidence)}")

        print(f"总共过滤掉: {len(evidence_pool) - len(filtered_evidence)} 条证据\n")

        return filtered_evidence

    def _rule_based_filter(self, evidence_pool: List[Dict]) -> List[Dict]:

        """Rule-based fast filtering"""

        filtered = []

        for evidence in evidence_pool:

            content = evidence.get("content", "").lower()

            title = evidence.get("title", "").lower()

            # 过滤规则

            skip = False

            # 1. 空内容

            if not content or len(content.strip()) < 20:
                skip = True

            # 2. 登录页面关键词

            login_keywords = [

                "please log in", "please login", "sign in to continue",

                "login to view", "登录查看", "请登录", "需要登录",

                "authentication required", "access denied"

            ]

            if any(kw in content[:200] for kw in login_keywords):
                skip = True

            # 3. error页面

            error_keywords = [

                "404", "page not found", "not found",

                "error 404", "页面不存在", "找不到页面"

            ]

            if any(kw in title or kw in content[:100] for kw in error_keywords):
                skip = True

            # 4. Cookie/隐私通知（如果整个内容都是关于这个）

            if "cookie" in content[:300] and len(content) < 500:
                skip = True

            if not skip:
                filtered.append(evidence)

        return filtered

    def _deduplicate_evidence(self, evidence_pool: List[Dict]) -> List[Dict]:

        """Remove duplicate evidence - based on URL and content similarity"""

        seen_urls = set()

        seen_content_hashes = set()

        deduped = []

        for evidence in evidence_pool:

            url = evidence.get("url", "")

            content = evidence.get("content", "")

            # 检查 URL 是否重复

            if url and url in seen_urls:
                continue

            # 检查内容是否高度相似（简单哈希）

            content_hash = hash(content[:500])  # 只比较前500字符

            if content_hash in seen_content_hashes:
                continue

            seen_urls.add(url)

            seen_content_hashes.add(content_hash)

            deduped.append(evidence)

        return deduped

    def _format_evidence_batch(self, batch: List[Dict]) -> str:

        """Format evidence batch for LLM input"""

        lines = []

        for i, evidence in enumerate(batch, 1):
            lines.append(f"Evidence {i}:")

            lines.append(f"  ID: {evidence['id']}")

            lines.append(f"  Title: {evidence.get('title', 'N/A')}")

            lines.append(f"  URL: {evidence.get('url', 'N/A')}")

            lines.append(f"  Content Preview: {evidence.get('content', '')[:300]}...")

            lines.append("")

        return "\n".join(lines)

    def _parse_filter_output(self, text: str) -> List[Dict]:

        """Parse LLM output filtering decisions"""

        import re

        # 尝试提取 JSON

        try:

            # 策略1: 直接解析

            parsed = json.loads(text)

            if 'filtered_evidence' in parsed:
                return parsed['filtered_evidence']

        except:

            pass

        # 策略2: 提取代码块

        code_block_match = re.search(

            r'```(?:json)?\s*(\{.*?"filtered_evidence".*?\})\s*```',

            text,

            re.DOTALL

        )

        if code_block_match:

            try:

                parsed = json.loads(code_block_match.group(1))

                if 'filtered_evidence' in parsed:
                    return parsed['filtered_evidence']

            except:

                pass

        # 策略3: 查找 JSON 对象

        json_match = re.search(

            r'\{[^{]*?"filtered_evidence"\s*:\s*\[.*?\]\s*\}',

            text,

            re.DOTALL

        )

        if json_match:

            try:

                parsed = json.loads(json_match.group(0))

                if 'filtered_evidence' in parsed:
                    return parsed['filtered_evidence']

            except:

                pass

        # 解析failed，默认保留所有

        print(f"  ⚠ 无法解析过滤输出，默认保留所有证据")

        return []