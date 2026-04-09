"""
ArgumentMerger - 论点合并器

将多个相似的论点合并为一个，并整合它们的支持证据
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import re
import json
from typing import List, Tuple

# 从mad导入共享组件
from utils.simple_prompt import SimplePromptTemplate as PromptTemplate
from utils.simple_chain import SimpleLLMChain as LLMChain

# 从mad_v2导入模型
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.models import ClaimPoint


class ArgumentMerger:
    """
    论点合并器

    核心功能：
    1. 检测哪些论点实质上表达了相同的观点
    2. 将相似论点合并为一个，整合所有支持证据
    """

    def __init__(self, llm):
        """
        初始化

        Args:
            llm: LangChain compatible LLM (QwenLLMWrapper)
        """
        self.llm = llm

        # 论点相似度检测Prompt
        self.similarity_template = PromptTemplate(
            input_variables=["point1_text", "point2_text", "sub_claim"],
            template="""You are a fact-checking expert. Determine if two claim points express the SAME essential assertion.

**Sub-Claim Context:**
{sub_claim}

**Claim Point 1:**
{point1_text}

**Claim Point 2:**
{point2_text}

**Task:**
Determine if these two claim points are expressing the SAME core assertion (even if worded differently).

**Criteria for "SAME":**
1. **Same factual content**: They assert the same fact (e.g., both say "X has 23 titles")
2. **Same scope**: Same qualifiers (e.g., both say "in Open Era" or both say "all-time")
3. **Compatible**: If both were true, they don't contradict each other

**Criteria for "DIFFERENT":**
1. Different facts (e.g., "23 titles" vs "24 titles")
2. Different scope (e.g., "all-time" vs "Open Era")
3. Opposite direction (one supports, one refutes)
4. Different aspects of the sub-claim

**Examples:**

Example 1 - SAME:
- Point 1: "Serena Williams won 23 Grand Slam singles titles"
- Point 2: "Williams has 23 Grand Slam championship wins in singles"
→ Same fact, same scope, just different wording

Example 2 - DIFFERENT:
- Point 1: "Serena Williams has 23 Grand Slam titles in the Open Era"
- Point 2: "Serena Williams has the most Grand Slam titles in all of tennis history"
→ Different scope ("Open Era" vs "all of history")

Example 3 - DIFFERENT:
- Point 1: "Serena Williams has 23 Grand Slam titles"
- Point 2: "Margaret Court has 24 Grand Slam titles"
→ Different subjects, different facts

**Output Format (JSON):**
{{
  "are_same": true,
  "confidence": 0.95,
  "rationale": "Both assert the same fact about 23 Grand Slam titles, just with different phrasing"
}}

OR

{{
  "are_same": false,
  "confidence": 0.9,
  "rationale": "Different scope: one refers to Open Era, the other to all-time history"
}}

Now analyze:"""
        )

        self.similarity_chain = LLMChain(llm=self.llm, prompt=self.similarity_template)

    def merge_similar_points(
        self,
        claim_points: List[ClaimPoint],
        similarity_threshold: float = 0.7
    ) -> List[ClaimPoint]:
        """
        合并相似的论点

        Args:
            claim_points: 原始论点列表
            similarity_threshold: 相似度阈值（0-1），超过此阈值认为是相同论点

        Returns:
            List[ClaimPoint]: 合并后的论点列表
        """
        if len(claim_points) <= 1:
            return claim_points

        print(f"\n[论点合并] 检测 {len(claim_points)} 个论点中的相似项...")

        # 禁用搜索
        original_search = self.llm.enable_search
        original_force = self.llm.force_search
        self.llm.enable_search = False
        self.llm.force_search = False

        try:
            # 按sub_claim分组（只合并针对同一个子claim的论点）
            groups_by_subclaim = {}
            for point in claim_points:
                key = point.sub_claim_id
                if key not in groups_by_subclaim:
                    groups_by_subclaim[key] = []
                groups_by_subclaim[key].append(point)

            merged_points = []

            # 对每个子claim的论点组进行合并
            for sub_claim_id, points in groups_by_subclaim.items():
                if len(points) == 1:
                    merged_points.extend(points)
                    continue

                print(f"  检测子Claim组 {sub_claim_id}: {len(points)} 个论点")

                # 合并这组论点
                merged_group = self._merge_point_group(points, similarity_threshold)
                merged_points.extend(merged_group)

                if len(merged_group) < len(points):
                    print(f"  ✓ 合并后: {len(merged_group)} 个论点（减少了 {len(points) - len(merged_group)} 个）")

            print(f"\n✓ 论点合并completed: {len(claim_points)} → {len(merged_points)}")
            return merged_points

        finally:
            # 恢复原始配置
            self.llm.enable_search = original_search
            self.llm.force_search = original_force

    def _merge_point_group(
        self,
        points: List[ClaimPoint],
        similarity_threshold: float
    ) -> List[ClaimPoint]:
        """
        合并一组论点（针对同一个子claim）

        使用贪心算法：
        1. 遍历每个论点
        2. 检查它是否与已有的合并组相似
        3. 如果相似，合并；否则创建新组
        """
        merged_groups = []  # List of ClaimPoint（每个是合并后的论点）

        for point in points:
            # 检查是否与现有组相似
            merged_into_existing = False

            for merged_point in merged_groups:
                # 检测相似度
                is_same, confidence = self._check_similarity(
                    point, merged_point, similarity_threshold
                )

                if is_same:
                    # 合并到现有组
                    self._merge_into(merged_point, point)
                    merged_into_existing = True
                    print(f"    ✓ 合并论点: {point.id} → {merged_point.id}")
                    break

            if not merged_into_existing:
                # 创建新组
                merged_groups.append(point)

        return merged_groups

    def _check_similarity(
        self,
        point1: ClaimPoint,
        point2: ClaimPoint,
        threshold: float
    ) -> Tuple[bool, float]:
        """
        检查两个论点是否相似

        Returns:
            (is_same, confidence)
        """
        try:
            result = self.similarity_chain.invoke({
                "point1_text": point1.point_text,
                "point2_text": point2.point_text,
                "sub_claim": point1.sub_claim_text
            })

            text = result.get('text', '')

            # 解析JSON输出
            parsed = self._parse_similarity_output(text)

            if parsed:
                are_same = parsed.get('are_same', False)
                confidence = float(parsed.get('confidence', 0.5))

                # 只有当confidence也超过阈值时才认为是相同的
                if are_same and confidence >= threshold:
                    return (True, confidence)

            return (False, 0.0)

        except Exception as e:
            print(f"    ⚠ 相似度检测failed: {e}")
            # failed时使用简单的文本相似度
            return self._simple_text_similarity(point1, point2, threshold)

    def _parse_similarity_output(self, text: str) -> dict:
        """Parse similarity detection output"""
        json_match = re.search(r'\{[^{}]*"are_same"[^{}]*\}', text, re.DOTALL)

        if json_match:
            try:
                json_str = json_match.group(0)
                json_str = json_str.replace("'", '"')
                parsed = json.loads(json_str)
                return parsed
            except:
                pass

        return {}

    def _simple_text_similarity(
        self,
        point1: ClaimPoint,
        point2: ClaimPoint,
        threshold: float
    ) -> Tuple[bool, float]:
        """
        简单的文本相似度（后备方案）
        基于Jaccard相似度
        """
        text1 = point1.point_text.lower()
        text2 = point2.point_text.lower()

        # Tokenize
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())

        # Jaccard相似度
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        if len(union) == 0:
            return (False, 0.0)

        similarity = len(intersection) / len(union)

        return (similarity >= threshold, similarity)

    def _merge_into(self, target: ClaimPoint, source: ClaimPoint):
        """
        将source论点合并到target论点中

        合并策略：
        1. 保留target的point_text（作为代表）
        2. 整合所有supporting_evidence
        3. 更新质量和置信度（取平均或最大值）
        4. 更新优先级（取最高）
        """
        # 整合证据ID
        target.supporting_evidence_ids.extend(source.supporting_evidence_ids)

        # 整合证据片段
        target.supporting_evidence_snippets.extend(source.supporting_evidence_snippets)

        # 整合来源
        target.source_urls.extend(source.source_urls)
        target.source_domains.extend(source.source_domains)
        
        # 整合发布时间
        target.evidence_published_times.extend(source.evidence_published_times)

        # 去重
        target.supporting_evidence_ids = list(set(target.supporting_evidence_ids))
        target.source_urls = list(set(target.source_urls))
        target.source_domains = list(set(target.source_domains))

        # 更新可信度（取最高）
        cred_map = {"High": 2, "Medium": 1, "Low": 0}
        target_cred_score = cred_map.get(target.credibility, 1)
        source_cred_score = cred_map.get(source.credibility, 1)

        if source_cred_score > target_cred_score:
            target.credibility = source.credibility
        
        # 更新权威性优先级（取最高）
        target_auth_score = cred_map.get(target.authority_priority, 1)
        source_auth_score = cred_map.get(source.authority_priority, 1)
        
        if source_auth_score > target_auth_score:
            target.authority_priority = source.authority_priority
        
        # 更新时效性优先级（取最高）
        target.timeliness_priority = max(target.timeliness_priority, source.timeliness_priority)

        # 更新质量和置信度（取平均）
        total_evidence_count = len(target.supporting_evidence_ids)
        target.quality_score = (target.quality_score + source.quality_score) / 2
        target.confidence = (target.confidence + source.confidence) / 2

        # 证据数量越多，置信度提升
        target.confidence = min(1.0, target.confidence * (1 + 0.1 * (total_evidence_count - 1)))
