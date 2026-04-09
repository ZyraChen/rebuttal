"""
ClaimBasedArgumentExtractor - 改进版论点提取器

改进点：
1. 修复了提示词的格式error
2. 更明确的任务描述和提取规则
3. 更清晰的置信度评分标准
4. 添加了具体示例
5. 强调忠实于原文，减少过度推理
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uuid
import json
import re
from typing import List, Dict, Optional
from datetime import datetime
from urllib.parse import urlparse

from utils.simple_prompt import SimplePromptTemplate as PromptTemplate
from utils.simple_chain import SimpleLLMChain as LLMChain

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.models import SubClaim, ClaimPoint


class ClaimBasedArgumentExtractorV2:
    """
    改进版论点提取器
    
    主要改进：
    - 更清晰的提示词结构
    - 明确的提取规则
    - 具体的评分标准
    - 减少过度推理
    """

    def __init__(self, llm):
        self.llm = llm

        # 改进的提取提示词
        self.extraction_template = PromptTemplate(
            input_variables=["claim", "sub_claim", "document_content", "document_title",
                           "document_url", "agent_type"],
            template="""You are a fact-checking assistant. Your task is to extract ONE specific argument from the provided evidence that addresses the given sub-claim question.

**Parent Claim:**
{claim}

**Sub-Claim Question:**
{sub_claim}

**Evidence Document:**
Title: {document_title}
URL: {document_url}
Content:
{document_content}

**Task:**
Extract ONE specific argument from this evidence that:
1. Directly answers or relates to the sub-claim question
2. Can SUPPORT or REFUTE the sub-claim (extract contradictions if present)
3. States what the evidence says, not what you think about it

**Extraction Rules:**
1. **Fidelity**: Extract ONLY what the evidence explicitly states
   - Quote or closely paraphrase the document
   - Don't infer, speculate, or add information not present
   
2. **Precision**: Verify all factual details
   - Dates, day-of-week, numbers, entity names
   - Distinguish between publication time vs event time
   - Preserve qualifiers (e.g., "around", "approximately", "at least")
   
3. **Stance**: Extract both supporting and contradicting evidence
   - If evidence supports sub-claim: extract the support
   - If evidence contradicts sub-claim: extract the contradiction
   - Be neutral - don't choose sides
   
4. **Relevance**: Only extract if directly relevant
   - Must address the sub-claim question
   - Must have sufficient detail (confidence ≥ 0.3)
   - If irrelevant or too vague: return {{"argument": null}}

**When to Return Null:**
- Evidence doesn't mention the sub-claim topic at all
- Evidence is too vague (no specific facts, just general statements)
- Confidence would be < 0.3
- Evidence is completely off-topic

**Argument Quality Requirements:**
Your extracted argument must be:
- **Specific**: Include concrete details (numbers, dates, names)
- **Faithful**: Directly from the document, not your interpretation
- **Relevant**: Clearly related to the sub-claim question
- **Contextual**: Include enough context to be understandable, 
              but don't add information not in the source

**Confidence Scoring:**
- **0.9-1.0**: Document explicitly states with specific details 
             (exact numbers, dates, clear statements)
- **0.7-0.9**: Document clearly states but with minor ambiguity
             (e.g., "around 100" vs "exactly 100")
- **0.5-0.7**: Document implies with good supporting context
- **0.3-0.5**: Document mentions but lacks specifics
- **< 0.3**: Too vague - DO NOT extract, return null

**Published Time Extraction:**
Look for the document's publication date (not event date):
- Explicit markers: "发布于", "Published:", "发表时间"
- Format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS
- If not found: return null

**Output Format (JSON):**
{{
  "argument": {{
    "argument_text": "冥王星的远日点距离太阳约74亿公里，近日点约44亿公里。",
    "supporting_snippet": "根据官方记录，冥王星轨道的远日点约为49天文单位（约74亿公里），近日点约30天文单位...",
    "confidence": 0.95,
    "addresses_sub_claim": true,
    "published_time": "2024-01-15"
  }}
}}

**OR if evidence is irrelevant:**
{{
  "argument": null
}}

**Examples:**

Example 1 - Good Extraction:
Sub-claim: "冥王星距离太阳最远约74亿公里"
Evidence: "冥王星的远日点约为49天文单位，即约74亿公里"
✓ Extract: argument_text: "冥王星的远日点距离太阳约74亿公里", confidence: 0.95

Example 2 - Contradiction (Still Extract):
Sub-claim: "威廉姆斯是历史上大满贯最多的选手"
Evidence: "威廉姆斯在公开赛时代拥有23个大满贯，而玛格丽特·考特在整个职业生涯中共获得24个"
✓ Extract: argument_text: "威廉姆斯在公开赛时代有23个大满贯，但考特在整个职业生涯中有24个", confidence: 0.9
(Note: This contradicts the claim - extract it anyway)

Example 3 - Too Vague (Return Null):
Sub-claim: "冥王星距离太阳最远约74亿公里"
Evidence: "冥王星是一颗距离太阳很远的矮行星"
✗ Return: {{"argument": null}}
(Reason: No specific distance mentioned)

Now analyze the evidence and extract the argument:"""
        )

        self.extraction_chain = LLMChain(llm=self.llm, prompt=self.extraction_template)

    def extract_points(
        self,
        claim: str,
        sub_claims: List[SubClaim],
        search_results: List[Dict],
        agent_type: str,
        round_num: int,
        search_query: str
    ) -> List[ClaimPoint]:
        """
        从搜索结果中提取论点
        
        使用改进的提示词，减少过度推理，提高提取准确性
        """
        claim_points = []

        # 临时启用搜索和思考模式
        original_search = self.llm.enable_search
        original_force = self.llm.force_search
        original_thinking = getattr(self.llm, 'enable_thinking', False)
        
        self.llm.enable_search = True
        self.llm.force_search = True
        if hasattr(self.llm, 'enable_thinking'):
            self.llm.enable_thinking = True

        try:
            # 对每个证据文档
            for doc_idx, doc in enumerate(search_results[:5]):  # 只Processing前5个结果
                doc_content = doc.get('content', doc.get('description', ''))
                doc_title = doc.get('title', '')
                doc_url = doc.get('url', '')

                if len(doc_content) < 100:
                    continue

                # 截取内容（避免过长）
                doc_content_truncated = doc_content[:1500]

                # 对每个子claim提问
                for sub_claim in sub_claims:
                    try:
                        # 调用LLM提取论点
                        result = self.extraction_chain.invoke({
                            "claim": claim,
                            "sub_claim": sub_claim.text,
                            "document_content": doc_content_truncated,
                            "document_title": doc_title,
                            "document_url": doc_url,
                            "agent_type": agent_type
                        })

                        text = result.get('text', '')

                        # 调试输出
                        print(f"\n  [V2 提取器] 文档 {doc_idx+1} + 子Claim '{sub_claim.text[:40]}...'")
                        print(f"  LLM原始输出: {text[:300]}...")

                        # 解析JSON输出
                        extracted = self._parse_extraction_output(text)

                        if not extracted:
                            print(f"  ❌ JSON解析failed")
                            continue

                        if extracted.get('argument') is None:
                            print(f"  ⚠ LLM返回 argument=null（文档不相关）")
                            continue

                        point_data = extracted['argument']

                        # 验证必需字段
                        if 'argument_text' not in point_data:
                            print(f"  ❌ 缺少 argument_text 字段")
                            continue

                        # 获取置信度
                        confidence = float(point_data.get('confidence', 0.5))

                        # Skip置信度太低的
                        if confidence < 0.3:
                            print(f"  ❌ 置信度过低: {confidence:.2f} < 0.3")
                            continue

                        # 评估可信度
                        credibility = self._assess_credibility(doc_url)

                        # 计算质量分数
                        quality = self._assess_quality(
                            point_data['argument_text'],
                            point_data.get('supporting_snippet', ''),
                            credibility
                        )

                        # 获取证据发布时间
                        published_time = point_data.get('published_time')
                        if not published_time:
                            published_time = doc.get('published_time')
                        if not published_time:
                            published_time = self._extract_time_from_content(doc_content)

                        # 计算时效性优先级
                        timeliness_priority = self._calculate_timeliness_priority(published_time)

                        # 创建ClaimPoint对象
                        claim_point = ClaimPoint(
                            id=f"{agent_type}_{round_num}_point_{uuid.uuid4().hex[:8]}",
                            point_text=point_data['argument_text'].strip(),
                            sub_claim_id=sub_claim.id,
                            sub_claim_text=sub_claim.text,
                            supporting_evidence_ids=[f"{doc_idx}_{doc_url}"],
                            supporting_evidence_snippets=[point_data.get('supporting_snippet', doc_content[:200])],
                            source_urls=[doc_url],
                            source_domains=[urlparse(doc_url).netloc or '未知'],
                            credibility=credibility,
                            retrieved_by=agent_type,
                            round_num=round_num,
                            timestamp=datetime.now(),
                            quality_score=quality,
                            confidence=confidence,
                            authority_priority=credibility,
                            timeliness_priority=timeliness_priority,
                            evidence_published_times=[published_time]
                        )

                        claim_points.append(claim_point)

                        print(f"  ✓ 提取论点: {claim_point.id}")
                        print(f"    论点: {claim_point.point_text[:80]}...")
                        print(f"    置信度: {confidence:.2f}, 质量: {quality:.2f}")

                    except Exception as e:
                        print(f"  ⚠ 论点提取failed: {e}")
                        continue

        finally:
            # 恢复原始配置
            self.llm.enable_search = original_search
            self.llm.force_search = original_force
            if hasattr(self.llm, 'enable_thinking'):
                self.llm.enable_thinking = original_thinking

        return claim_points

    # 以下方法与原版相同，为简洁起见省略
    # 包括：_parse_extraction_output, _clean_json, _assess_credibility, 
    #      _assess_quality, _calculate_timeliness_priority, _extract_time_from_content
    
    def _parse_extraction_output(self, text: str) -> Dict:
        """Parse JSON format from LLM output"""
        # 策略1: 直接解析
        try:
            parsed = json.loads(text)
            if 'argument' in parsed:
                return parsed
        except:
            pass

        # 策略2: 提取代码块
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?"argument".*?\})\s*```', text, re.DOTALL)
        if code_block_match:
            try:
                json_str = self._clean_json(code_block_match.group(1))
                parsed = json.loads(json_str)
                if 'argument' in parsed:
                    return parsed
            except:
                pass

        # 策略3: 查找JSON对象
        json_patterns = [
            r'\{\s*"argument"\s*:\s*\{[^}]*\}\s*\}',
            r'\{\s*"argument"\s*:\s*null\s*\}',
        ]

        for pattern in json_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    json_str = self._clean_json(match.group(0))
                    parsed = json.loads(json_str)
                    if 'argument' in parsed:
                        return parsed
                except:
                    continue

        return {}

    def _clean_json(self, json_str: str) -> str:
        """Clean JSON string"""
        json_str = json_str.replace("'", '"')
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        return json_str.strip()

    def _assess_credibility(self, url: str) -> str:
        """Evaluate source credibility"""
        domain = urlparse(url).netloc.lower()
        
        high_cred = ['gov', 'edu', 'who.int', 'wikipedia.org', 'nature.com', 
                     'science.org', 'reuters.com', 'bbc.com', 'un.org']
        
        for keyword in high_cred:
            if keyword in domain:
                return "High"
        
        if any(ext in domain for ext in ['com', 'org', 'net']):
            return "Medium"
        
        return "Low"

    def _assess_quality(self, point_text: str, evidence: str, credibility: str) -> float:
        """Evaluate argument quality"""
        cred_score = {"High": 1.0, "Medium": 0.6, "Low": 0.3}.get(credibility, 0.5)
        return min(1.0, cred_score)

    def _calculate_timeliness_priority(self, published_time: Optional[str]) -> int:
        """Calculate timeliness priority"""
        if not published_time:
            return 0
        
        try:
            from dateutil import parser
            from dateutil.tz import UTC
            pub_date = parser.parse(published_time)
            
            if pub_date.tzinfo is not None:
                now = datetime.now(UTC)
            else:
                now = datetime.now()
            
            if pub_date.tzinfo is not None and now.tzinfo is None:
                pub_date = pub_date.astimezone().replace(tzinfo=None)
            
            days_ago = (now - pub_date).days
            
            if days_ago < 0:
                return 3
            elif days_ago <= 365:
                return 3
            elif days_ago <= 365 * 3:
                return 2
            else:
                return 1
        except:
            return 0

    def _extract_time_from_content(self, content: str) -> Optional[str]:
        """Extract time from content"""
        if not content:
            return None
        
        search_text = content[:500] + content[-500:] if len(content) > 1000 else content
        
        date_patterns = [
            r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)',
            r'(\d{4}年\d{1,2}月\d{1,2}日)',
            r'(发布于\s*[:：]?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2}))',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                date_str = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group(0)
                try:
                    date_str = date_str.replace('年', '-').replace('月', '-').replace('日', '').replace('/', '-')
                    parts = date_str.split('-')
                    if len(parts) == 3:
                        year, month, day = parts
                        if len(year) == 4 and 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
                            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                except:
                    continue
        
        return None

