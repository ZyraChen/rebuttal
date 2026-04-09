"""
ClaimBasedArgumentExtractor - 基于子Claim的论点提取器

对每个证据，用子claims作为问题去提问，总结得出论点
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

# 从mad导入共享组件
from utils.simple_prompt import SimplePromptTemplate as PromptTemplate
from utils.simple_chain import SimpleLLMChain as LLMChain

# 从mad_v2导入模型
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.models import SubClaim, ClaimPoint


class ClaimBasedArgumentExtractor:
    """
    基于子Claim的论点提取器

    核心流程：
    1. 对每个证据（搜索结果）
    2. 用每个子claim作为问题去提问这个证据
    3. LLM总结得出一个论点（对该子claim的回答）
    """

    def __init__(self, llm):
        """
        初始化

        Args:
            llm: LangChain compatible LLM (QwenLLMWrapper)
        """
        self.llm = llm

        # 提取论点的Prompt
        self.extraction_template = PromptTemplate(
            input_variables=["claim","sub_claim", "document_content", "document_title",
                           "document_url", "agent_type"],
#             template="""You are a fact-checking assistant. Your task is to extract one specific argument from the provided evidence that addresses the given sub-claim question.
# **Parent Claim:**
# {claim}
# **Sub-Claim Question:**
# {sub_claim}
#
# **Evidence Document:**
# Title: {document_title}
# URL: {document_url}
# Content:
# {document_content}
#
# **Task:** Extract ONE specific argument from this evidence that addresses the sub-claim.
#
# **Critical Rules:**
# - Extract ONLY what the evidence explicitly states - do not speculate
# - Check ALL details (dates, day-of-week, numbers, entity names)
# - If evidence contradicts the sub-claim, extract the contradiction as the argument
# - Note the distinction between the time of information release and the time of event occurrence.
# - If evidence is irrelevant or doesn't address the sub-claim, output:
#   {{"claim_point": null}}
#
# **ARGUMENT Requirements:**
# 1. **Specific**: State a concise, clear, specific assertion (not vague)
# 2. **Evidence-based**: Directly quote or paraphrase arguments from the document
# 3. **Relevant**: the evidence can address this subclaim
# 4. **Standalone**: The ARGUMENT should be understandable and reasonable on its own
#
# **Published Time Extraction:**
# Extract the publication/publish date of this document. Look for:
# - Explicit dates like "发布于2024年1月1日", "Published: 2024-01-01", "2024年1月1日发布"
# - Dates in article headers, footers, or metadata
# - Dates mentioned in the first or last paragraph
# - Format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS
# - If no date found, return null
#
# **Confidence Scoring:**
# - **1.0**: Document explicitly states this fact with clear evidence
# - **0.6-0.9**: Document strongly implies this, with good supporting details
# - **0.3-0.6**: Document mentions this but with limited details
# - **<0.3**: Don't extract - relevance too low
#
# **Output Format (JSON):**
# {{
#   "claim_point": {{
#     "point_text": "在公开赛时代，塞雷娜·威廉姆斯已斩获23项大满贯单打冠军头衔。",
#     "supporting_snippet": "根据官方记录，威廉姆斯在2017年澳大利亚网球公开赛上斩获个人Round23座大满贯奖杯...",
#     "confidence": 0.95,
#     "addresses_sub_claim": true,
#     "published_time": "2024-01-15" 或 null
#   }}
# }}
#
# Now analyze the evidence and extract the ARGUMENT:"""
                        template="""You are a fact-checking assistant. Your task is to extract one specific argument from the provided evidence that addresses the given sub-claim question.
            **Parent Claim:**
            {claim}
            **Sub-Claim Question:**
            {sub_claim}

            **Evidence Document:**
            Title: {document_title}
            URL: {document_url}
            Content:
            {document_content}

            *Task:**
            Extract ONE specific argument from this evidence that directly addresses the sub-claim question. If the evidence does not contain relevant information, return null as specified below.
            **Extraction Rules:**
            1. Extract only what the evidence explicitly states. Do not infer, speculate, or add information not present in the document.
            2. Verify all factual details including dates, day-of-week, numbers, and entity names against the source text.
            3. If the evidence contradicts the sub-claim, extract the contradicting information as your argument.
            4. Distinguish between when information was published and when events occurred. Prioritize extracting the event time if both are present.
            5. If evidence is irrelevant or insufficient to address the sub-claim, return: {{"argument": null}}

            **Argument Quality Standards:**
            Your extracted argument must be:
            - Specific and concrete, avoiding vague generalizations
            - Directly quoted or closely paraphrased from the document with accurate details
            - Clearly relevant to answering the sub-claim question
            - Self-contained and understandable without requiring additional context
            - If evidence is irrelevant or doesn't address the sub-claim, output:
              {{"argument": null}}

            **Published Time Extraction:**
            Extract the publication/publish date of this document. Look for:
            - Explicit dates like "发布于2024年1月1日", "Published: 2024-01-01", "2024年1月1日发布"
            - Dates in article headers, footers, or metadata
            - Dates mentioned in the first or last paragraph
            - Format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS
            - If no date found, return null

            **Confidence Scoring:**
            - **1.0**: Document explicitly states this fact with clear evidence
            - **0.6-0.9**: Document strongly implies this, with good supporting details
            - **0.3-0.6**: Document mentions this but with limited details
            - **<0.3**: Don't extract - relevance too low

            **Output Format (JSON):**
            {{
              "argument": {{
                "argument_text": "在公开赛时代，塞雷娜·威廉姆斯已斩获23项大满贯单打冠军头衔。",
                "supporting_snippet": "根据官方记录，威廉姆斯在2017年澳大利亚网球公开赛上斩获个人Round23座大满贯奖杯...",
                "confidence": 0.95,
                "addresses_sub_claim": true,
                "published_time": "2024-01-15" 或 null
              }}
            }}

            Now analyze the evidence and extract the ARGUMENT:"""
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

        Args:
            sub_claims: 子claim列表
            search_results: Jina搜索结果列表
            agent_type: "pro" 或 "con"
            round_num: 当前轮次
            search_query: 使用的搜索词

        Returns:
            List[ClaimPoint]: 提取的论点列表
        """
        claim_points = []

        # 禁用搜索（这是分析任务）
        original_search = self.llm.enable_search
        original_force = self.llm.force_search
        self.llm.enable_search = True
        self.llm.force_search = True
        self.llm.enable_thinking= True


        try:
            # 对每个证据
            for doc_idx, doc in enumerate(search_results[:5]):  # 只Processing前5个结果
                doc_content = doc.get('content', doc.get('description', ''))
                doc_title = doc.get('title', '')
                doc_url = doc.get('url', '')

                if len(doc_content) < 100:
                    continue

                # 截取内容
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

                        # 【调试】显示LLM原始输出
                        print(f"\n  [调试] 文档 {doc_idx+1} + 子Claim '{sub_claim.text[:40]}...'")
                        print(f"  LLM原始输出: {text[:300]}...")

                        # 解析JSON输出
                        extracted = self._parse_extraction_output(text)

                        if not extracted:
                            print(f"  ❌ JSON解析failed，extracted为空")
                            continue

                        if extracted.get('argument') is None:
                            print(f"  ⚠ LLM返回 argument=null（文档不相关）")
                            continue

                        point_data = extracted['argument']

                        # 验证必需字段
                        if 'argument_text' not in point_data:
                            print(f"  ❌ 缺少 argument_text 字段: {point_data}")
                            continue

                        # 获取置信度
                        confidence = float(point_data.get('confidence', 0.5))

                        # Skip置信度太低的
                        if confidence < 0.1:
                            print(f"  ❌ 置信度过低: {confidence:.2f} < 0.1")
                            continue

                        # 评估可信度
                        credibility = self._assess_credibility(doc_url)

                        # 计算质量分数
                        quality = self._assess_quality(
                            point_data['argument_text'],
                            point_data.get('supporting_snippet', ''),
                            credibility
                        )

                        # 获取证据发布时间（优先使用 LLM 提取的）
                        published_time = point_data.get('published_time')  # LLM 提取的时间
                        
                        # 如果 LLM 没有提取到，回退到原始数据
                        if not published_time:
                            published_time = doc.get('published_time')
                        
                        # 如果还是没有，尝试从文档内容中提取（作为最后手段）
                        if not published_time:
                            published_time = self._extract_time_from_content(doc_content)
                        
                        # 计算时效性优先级
                        timeliness_priority = self._calculate_timeliness_priority(published_time)
                        
                        # 记录时间来源（用于调试）
                        if published_time:
                            if point_data.get('published_time'):
                                print(f"    发布时间（LLM提取）: {published_time}")
                            elif doc.get('published_time'):
                                print(f"    发布时间（原始数据）: {published_time}")
                            else:
                                print(f"    发布时间（内容提取）: {published_time}")

                        # 创建ClaimPoint对象
                        claim_point = ClaimPoint(
                            id=f"{agent_type}_{round_num}_point_{uuid.uuid4().hex[:8]}",
                            point_text=point_data['argument_text'].strip(),
                            sub_claim_id=sub_claim.id,
                            sub_claim_text=sub_claim.text,
                            supporting_evidence_ids=[f"{doc_idx}_{doc_url}"],  # 临时ID
                            supporting_evidence_snippets=[point_data.get('supporting_snippet', doc_content[:200])],
                            source_urls=[doc_url],
                            source_domains=[urlparse(doc_url).netloc or '未知'],
                            credibility=credibility,
                            retrieved_by=agent_type,
                            round_num=round_num,
                            timestamp=datetime.now(),
                            quality_score=quality,
                            confidence=confidence,
                            # 新增：双优先级系统
                            authority_priority=credibility,  # 权威性优先级与credibility一致
                            timeliness_priority=timeliness_priority,
                            evidence_published_times=[published_time]
                        )

                        claim_points.append(claim_point)

                        print(f"  ✓ 提取论点: {claim_point.id}")
                        print(f"    子Claim: {sub_claim.text}")
                        print(f"    论点: {claim_point.point_text[:80]}...")
                        print(f"    置信度: {confidence:.2f}, 质量: {quality:.2f}")

                    except Exception as e:
                        print(f"  ⚠ 论点提取failed (文档 {doc_idx+1}, 子claim: {sub_claim.text[:30]}...): {e}")
                        continue

        finally:
            # 恢复原始配置
            self.llm.enable_search = original_search
            self.llm.force_search = original_force

        return claim_points

    def _parse_extraction_output(self, text: str) -> Dict:
        """Parse JSON format from LLM output - using multiple strategies"""

        # 策略1: 直接解析整个文本
        try:
            parsed = json.loads(text)
            if 'argument' in parsed:
                return parsed
        except:
            pass

        # 策略2: 提取代码块中的JSON
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?"argument".*?\})\s*```', text, re.DOTALL)
        if code_block_match:
            try:
                json_str = self._clean_json(code_block_match.group(1))
                parsed = json.loads(json_str)
                if 'argument' in parsed:
                    return parsed
            except Exception as e:
                print(f"  ⚠ 代码块JSON解析failed: {e}")

        # 策略3: 查找嵌套JSON对象（支持claim_point内的嵌套）
        # 使用更灵活的正则 - 允许嵌套的大括号
        json_patterns = [
            r'\{\s*"argument"\s*:\s*\{[^}]*\}\s*\}',  # 简单嵌套
            r'\{\s*"argument"\s*:\s*null\s*\}',       # null情况
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

        # 策略4: 暴力查找（逐步扩展匹配范围）
        try:
            # 找到 "claim_point" 的位置，向前找最近的{，向后找匹配的}
            idx = text.find('"argument"')
            if idx != -1:
                # 向前找最近的 {
                start = text.rfind('{', 0, idx)
                if start != -1:
                    # 简单的括号匹配
                    brace_count = 0
                    for i in range(start, len(text)):
                        if text[i] == '{':
                            brace_count += 1
                        elif text[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_str = text[start:i+1]
                                try:
                                    parsed = json.loads(self._clean_json(json_str))
                                    if 'argument' in parsed:
                                        return parsed
                                except:
                                    pass
                                break
        except:
            pass

        print(f"  ⚠ 所有JSON解析策略failed")
        print(f"  原始文本: {text[:500]}...")
        
        # 尝试最后一次修复：查找并修复明显的格式error
        try:
            # 查找 claim_point 对象
            start_idx = text.find('"argument"')
            if start_idx != -1:
                # 尝试修复常见的格式error
                fixed_text = text
                
                # 修复：argument_text: 文本" -> argument_text": "文本"
                def fix_point_text(m):
                    key = m.group(1)
                    value = m.group(2).replace('"', '\\"')
                    ending = m.group(3)
                    return f'{key}: "{value}"{ending}'
                
                fixed_text = re.sub(
                    r'("argument_text")\s*:\s*([^",{\[\s][^,}]*?)([,}])',
                    fix_point_text,
                    fixed_text
                )
                
                # 再次尝试解析
                for strategy in [
                    lambda t: json.loads(t),
                    lambda t: json.loads(self._clean_json(t)),
                ]:
                    try:
                        parsed = strategy(fixed_text)
                        if 'argument' in parsed:
                            print(f"  ✓ 通过修复后解析successful")
                            return parsed
                    except:
                        continue
        except:
            pass
        
        return {}

    def _clean_json(self, json_str: str) -> str:
        """Clean JSON string and fix common format errors"""
        original = json_str
        
        # 1. 修复单引号
        json_str = json_str.replace("'", '"')
        
        # 2. 移除尾随逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # 3. 移除注释
        json_str = re.sub(r'//.*?\n', '\n', json_str)
        
        # 4. 修复缺少引号的字符串值（关键修复）
        # 匹配：": 后面跟着中文字符或非JSON特殊字符的文本
        # 例如："argument_text": 文本内容"  -> "argument_text": "文本内容"
        def fix_unquoted_string(match):
            key = match.group(1)
            value = match.group(2).strip()
            ending = match.group(3)
            
            # 如果value已经是引号包围的，不Processing
            if value.startswith('"') and value.endswith('"'):
                return match.group(0)
            
            # 如果value是数字、布尔值或null，不Processing
            if value.lower() in ['true', 'false', 'null'] or value.replace('.', '').replace('-', '').isdigit():
                return match.group(0)
            
            # 转义value中的引号
            value_escaped = value.replace('"', '\\"')
            return f'{key}: "{value_escaped}"{ending}'
        
        # 匹配模式：key": value, 或 key": value}
        json_str = re.sub(
            r'("(?:argument_text|supporting_snippet|published_time)")\s*:\s*([^",{\[\]}\s][^,}]*?)([,}])',
            fix_unquoted_string,
            json_str,
            flags=re.MULTILINE
        )
        
        # 5. 修复布尔值和null（如果被error地加了引号）
        json_str = re.sub(r':\s*"true"', ': true', json_str)
        json_str = re.sub(r':\s*"false"', ': false', json_str)
        json_str = re.sub(r':\s*"null"', ': null', json_str)
        
        # 6. 修复 addresses_sub_claim 字段（布尔值）
        json_str = re.sub(r'("addresses_sub_claim")\s*:\s*"([^"]+)"', 
                         lambda m: f'{m.group(1)}: {m.group(2).lower()}' if m.group(2).lower() in ['true', 'false'] else m.group(0),
                         json_str)
        
        # 7. 修复 confidence 字段（数字，如果被加了引号）
        json_str = re.sub(r'("confidence")\s*:\s*"([0-9.]+)"', r'\1: \2', json_str)
        
        return json_str
        return json_str.strip()

    def _assess_credibility(self, url: str) -> str:
        """Evaluate source credibility"""
        domain = urlparse(url).netloc.lower()

        high_cred_domains = [
            'gov', 'edu', 'who.int', 'wikipedia.org',
            'nature.com', 'science.org', 'reuters.com',
            'bbc.com', 'cnn.com', 'nytimes.com', 'un.org',
            'theguardian.com', 'apnews.com', 'npr.org'
        ]

        for keyword in high_cred_domains:
            if keyword in domain:
                return "High"

        if any(ext in domain for ext in ['com', 'org', 'net']):
            return "Medium"

        return "Low"

    def _assess_quality(self, point_text: str, evidence: str, credibility: str) -> float:
        """Evaluate argument quality"""
        # 基础分数来自可信度
        cred_score = {"High": 1.0, "Medium": 0.6, "Low": 0.3}.get(credibility, 0.5)

        # # 长度分数
        # point_len = len(point_text)
        # if 30 <= point_len <= 200:
        #     length_score = 1.0
        # elif point_len < 30:
        #     length_score = point_len / 30.0
        # else:
        #     length_score = max(0.5, 1.0 - (point_len - 200) / 300)
        #
        # # 证据详细程度
        # evidence_len = len(evidence)
        # evidence_score = min(1.0, evidence_len / 150)

        # 综合质量分数
        quality = cred_score

        return min(1.0, quality)
    
    def _calculate_timeliness_priority(self, published_time: Optional[str]) -> int:
        """
        计算时效性优先级
        
        Args:
            published_time: 发布时间字符串（可能为None或空字符串）
            
        Returns:
            0: 无时间信息
            1: 3年前或更早
            2: 1-3年前
            3: 1年内
        """
        if not published_time:
            return 0
        
        try:
            from dateutil import parser
            from dateutil.tz import UTC
            pub_date = parser.parse(published_time)
            
            # 统一时区Processing：如果pub_date有时区信息，now也要有时区
            # 如果pub_date没有时区，假设是本地时间
            if pub_date.tzinfo is not None:
                # pub_date有时区，使用UTC的now
                now = datetime.now(UTC)
            else:
                # pub_date没有时区，使用本地时间的now
                now = datetime.now()
            
            # 如果pub_date有时区而now没有，转换pub_date到本地时间
            if pub_date.tzinfo is not None and now.tzinfo is None:
                pub_date = pub_date.astimezone().replace(tzinfo=None)
            
            # 计算时间差（天数）
            days_ago = (now - pub_date).days
            
            if days_ago < 0:
                # 未来日期（可能是error），视为最新
                return 3
            elif days_ago <= 365:  # 1年内
                return 3
            elif days_ago <= 365 * 3:  # 1-3年
                return 2
            else:  # 3年以上
                return 1
                
        except Exception as e:
            # 解析failed，视为无时间信息
            print(f"  ⚠ 时间解析failed: {published_time} - {e}")
            return 0
    
    def _extract_time_from_content(self, content: str) -> Optional[str]:
        """
        从内容中提取时间（最后的回退方案）
        使用简单的正则表达式匹配常见日期格式
        
        Args:
            content: 文档内容
            
        Returns:
            时间字符串（YYYY-MM-DD格式）或 None
        """
        if not content:
            return None
        
        # 限制搜索范围（前500和后500字符，通常时间在开头或结尾）
        search_text = content[:500] + content[-500:] if len(content) > 1000 else content
        
        # 常见日期格式的正则表达式
        date_patterns = [
            r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)',  # YYYY-MM-DD, YYYY/MM/DD, YYYY年MM月DD日
            r'(\d{4}年\d{1,2}月\d{1,2}日)',              # 中文日期
            r'(发布于\s*[:：]?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2}))',  # 发布于: 2024-01-01
            r'(发布时间\s*[:：]?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2}))',  # 发布时间: 2024-01-01
            r'(Published\s*[:：]?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2}))',  # Published: 2024-01-01
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                # 优先使用捕获组，如果没有则使用整个匹配
                date_str = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group(0)
                
                # 标准化格式
                try:
                    # Processing中文日期
                    date_str = date_str.replace('年', '-').replace('月', '-').replace('日', '')
                    date_str = date_str.replace('/', '-')
                    
                    # 验证格式
                    parts = date_str.split('-')
                    if len(parts) == 3:
                        year, month, day = parts
                        if len(year) == 4 and 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
                            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                except:
                    continue
        
        return None
