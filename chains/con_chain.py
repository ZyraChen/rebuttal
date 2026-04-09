"""
Con Agent Query Chain - 生成搜索查询词
使用 LangChain 来组织 Prompt + LLM + Output Parsing
"""

from typing import List
from utils.simple_prompt import SimplePromptTemplate as PromptTemplate
from utils.simple_chain import SimpleLLMChain as LLMChain


class QueryOutputParser:
    """解析 LLM 输出为查询词列表"""

    def parse(self, text: str) -> List[str]:
        """解析输出"""
        queries = []
        for line in text.split('\n'):
            line = line.strip()
            if line and len(line) > 5:
                cleaned = line.lstrip('0123456789.、）)- *#').strip()
                if cleaned and len(cleaned) > 10:
                    queries.append(cleaned)

        return queries[:3]  # 最多返回3queries


class ConQueryChain:
    """
    Con Agent 查询生成链

    职责：根据当前状态生成反驳 claim 的搜索查询
    """

    def __init__(self, llm):
        """
        初始化

        Args:
            llm: LangChain compatible LLM (QwenLLMWrapper)
                 应配置为 enable_search=True, force_search=False
        """
        self.llm = llm

        # Round一轮的Prompt - 需要分解claim并生成多queries
        self.first_round_template = PromptTemplate(
            input_variables=["claim"],
            template="""It is now 3 January 2026. You are a fact-checking expert for the opposing side, responsible for finding refuting evidence for the following claim.
Claim: {claim}

Task objective:
Analyze this claim and identify 2-3 key verification points that need to be checked to refute it. For each verification point, generate one precise search query.

Analysis steps:
1. Identify the key assertions in this claim (e.g., key events, entities, numbers, timeframes, qualifiers like "most", "first", "only")
2. Determine which aspects are vulnerable or questionable for refuting the claim
3. For each critical verification point, generate a targeted search query

Query generation requirements:
1. Precision: Each query should target ONE specific verification point of the claim
2. Conciseness: Keep queries SHORT and focused on keywords (max 50 characters for Chinese, 100 for English)
3. Authority: Prioritize locating credible sources such as official websites, government agencies, academic journals, authoritative media, etc.
4. Diversity: Queries should cover different aspects for refutation (e.g., one for fact-checking the main event, one for verifying key entities, one for challenging specific details)
5. Format: Use keywords or short phrases, NOT full questions

Query format specifications (CRITICAL - Follow strictly):
- Chinese queries: Use concise keyword phrases, max 30-50 characters
  Good: `2024、2025年拉塞尔F1阿布扎比大奖赛Round五名`
- English queries: Use keywords connected with plus signs (+), max 100 characters
  Good: `Federer+Grand+Slam+titles+all+time+ranking`

Output requirements:
Output 2-3 SHORT search queries, one per line. Each query should be CONCISE.
Do not include query numbers, question marks, or extra text.
The language used for the query should be consistent with the claim.

Now please provide the queries (2-3 queries, one per line):"""
        )

        # 后续轮次的Prompt - 针对对手攻击进行反击
        self.followup_round_template = PromptTemplate(
            input_variables=["claim", "round_num", "opponent_summary", "existing_topics"],
            template="""It is now 3 January 2026. You are a fact-checking expert for the opposing side, responsible for finding refuting evidence for the following claim.
Claim: {claim}

Current round: Round {round_num}
{opponent_summary}
{existing_topics}

Task objective:
Generate 1 precise search query to find authoritative evidence in Jina Search to refute this claim.

Query generation requirements:
1. Precision: The query should directly correspond to the core content of the claim
2. Conciseness: Keep query SHORT and focused on keywords (max 50 characters for Chinese, 100 for English)
3. Authority: Prioritize locating credible sources
4. Targeting: If affirmative arguments exist, target their weak points
5. Differentiation: Avoid repetition with already searched topics

Query format specifications (CRITICAL - Follow strictly):
- Chinese queries: Use concise keyword phrases, max 30-50 characters
  Good: `纳达尔德约科维奇大满贯记录`
- English queries: Use keywords with plus signs (+), max 100 characters
  Good: `Nadal+Djokovic+Grand+Slam+record+comparison`

Output requirements:
Only output 1 SHORT search query (keywords/short phrase).
Do not include question marks, or extra text.
The language used for the query should be consistent with the claim.

Now please provide the query:"""
        )

        # 创建两个Chain
        self.first_round_chain = LLMChain(
            llm=self.llm,
            prompt=self.first_round_template,
            output_parser=QueryOutputParser()
        )

        self.followup_round_chain = LLMChain(
            llm=self.llm,
            prompt=self.followup_round_template,
            output_parser=QueryOutputParser()
        )

    def generate_queries(
        self,
        claim: str,
        round_num: int,
        opponent_evidences: List = None,
        existing_queries: List[str] = None
    ) -> List[str]:
        """
        生成搜索查询

        Args:
            claim: 要核查的 claim
            round_num: 当前轮次
            opponent_evidences: 支持方的证据列表
            existing_queries: 已有的查询（避免重复）

        Returns:
            查询词列表
        """
        # Round一轮：分解claim并生成多queries
        if round_num == 1:
            try:
                # Round一轮启用thinking模式，帮助LLM更好地分解claim和设计查询
                result = self.first_round_chain.invoke(
                    inputs={"claim": claim},
                    llm_kwargs={"enable_thinking": True}
                )

                # LangChain 返回字典，取 'text' 字段
                if isinstance(result, dict):
                    queries = result.get('text', [])
                else:
                    queries = result

                print(f"  [ConRound1轮] 基于claim分解生成了 {len(queries)} queries（使用thinking模式）")
                return queries[:3]  # Round一轮最多返回3queries

            except Exception as e:
                print(f"  ⚠ Con Chain (Round1轮) 调用failed: {e}")
                return []

        # 后续轮次：针对对手攻击进行反击
        else:
            # 构建对手论证摘要
            opponent_summary = ""
            if opponent_evidences:
                opponent_summary = "支持方最新论证:\n"
                for i, ev in enumerate(opponent_evidences[-3:], 1):
                    opponent_summary += f"{i}. [{ev.source}] {ev.content[:150]}...\n"

            # 已有主题
            existing_topics = ""
            if existing_queries:
                existing_topics = "已搜索过的主题(请避免重复):\n" + "\n".join([f"- {q}" for q in existing_queries[-5:]])

            # 调用 Chain
            try:
                result = self.followup_round_chain.invoke({
                    "claim": claim,
                    "round_num": round_num,
                    "opponent_summary": opponent_summary,
                    "existing_topics": existing_topics
                })

                if isinstance(result, dict):
                    queries = result.get('text', [])
                else:
                    queries = result

                # 过滤重复
                if existing_queries:
                    queries = [q for q in queries if q not in existing_queries]

                return queries[:1]  # 后续轮次只返回1queries

            except Exception as e:
                print(f"  ⚠ Con Chain (Round{round_num}轮) 调用failed: {e}")
                return []
