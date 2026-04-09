"""
Judge Chain - 生成最终Verdict
使用 LangChain 来组织 Prompt + LLM + Output Parsing
"""

from typing import List
from utils.simple_prompt import SimplePromptTemplate as PromptTemplate
from utils.simple_chain import SimpleLLMChain as LLMChain
from utils.models import Verdict


class JudgeChain:
    """
    Judge Verdict生成链

    职责：根据论辩图和被接受的证据生成最终Verdict
    """

    def __init__(self, llm):
        """
        初始化

        Args:
            llm: LangChain compatible LLM (QwenLLMWrapper)
        """
        self.llm = llm



        # 生成Verdict的 Prompt（改进版，明确区分Refuted和NEE）
        self.verdict_template = PromptTemplate(
            input_variables=["claim", "accepted_arguments", "all_arguments",
                           "attack_relation"],
            template="""You are an impartial fact-checking judge. You can use web search. Your task is to adjudicate the following claim.

Claim: "{claim}"

**Verdict Definition**:
1. **Refuted**: Use when the claim is refuted
2. **Supported**: Use when the claim is supported
3. **Not Enough Evidence** - Use only when:
   - the unverified issue involves unresolved academic controversy and Unverifiable subjective opinions
   - The macro perspective and philosophical perspectives that is unverifiable or unobservable
   - Speculation about the future

We have extracted the searched evidence into arguments and constructed an abstract argumentation framework. 
**Accepted Arguments from Argumentation Framework**:
{accepted_arguments}


**Judgment Protocol**:
Step 1: Decompose the claim into sub-claims. Identify all factual assertions that need verification.

Step 2: Map accepted arguments to sub-claims. For each sub-claim, determine whether the accepted arguments provide sufficient evidence. Mark each as either "covered" or "uncovered".

Step 3: Address evidence gaps. For sub-claims marked as "uncovered", conduct web searches or use your knowledge base to find relevant information. Also search if evidence pool has clear internal contradictions needing external verification.

Step 4: Apply critical evaluation rules:
    1. **Qualifier Contradictions**:
   - Note whether there is a contradiction in the scope of the determiner.
   - Example: Claim "Williams has most Grand Slams in history", Evidence "Williams has most in Open Era" → REFUTED (history ≠ Open Era)

    2. **Self-Consistency**:
   - Do NOT Rationalize Contradictions: If you notice a discrepancy between claim and evidence, acknowledge it honestly
   - Do NOT mental gymnastics to force Supported verdict when contradictions exist
   - Re-read claim and evidence carefully - does evidence ACTUALLY say what claim asserts?
   - Trust the evidence text literally, not your interpretation of "what they probably meant"
   - Interpret claim verbs in their NATURAL meaning, not most restrictive meaning
   - Do not deliberately confuse different concepts in order to disprove a claim.
   
    3. **Burden of Proof for Unverifiable Claims**:
   - For positive assertions (X happened, Y exists): If no evidence found → Refuted (burden on claimant)
   - For negative assertions (X didn't happen, no plagiarism): If no evidence of X → Supported
   - Example: Claim "Park Jisoo played Dewey", No evidence found → Refuted (positive claim needs proof)
   - Example: Claim "No plagiarism detected", No plagiarism evidence found → Supported (negative claim)
   - **Key distinction**: "Cannot find evidence of claim" = Refuted, "Cannot determine truth" = NEE

Step 5: Synthesize all evidence and determine verdict. Integrate accepted arguments with any additional information to assess each sub-claim and reach the final verdict.

Output format (JSON):
{{
  "justification": "完整的中文推理过程，需包含以下内容：首先分解claim为具体的子论点；然后分析accepted arguments对各子论点的覆盖情况，说明哪些论点已被充分证明，哪些存在证据缺口；如存在证据缺口，说明通过搜索或知识库补充了什么信息；明确检查并说明是否存在限定词矛盾、证据与声明的字面不一致、或举证责任问题；最后综合所有证据对每个子论点进行判断并得出最终verdict，确保逻辑严密无矛盾",
  "verdict": "...",
  "key_evidence": ["关键证据ID列表"]
}}

Now make the verdict:"""
        )

        self.verdict_chain = LLMChain(llm=self.llm, prompt=self.verdict_template)

    # def determine_stance(self, claim: str, evidence) -> str:
    #     """
    #     判断证据的立场
    #
    #     Args:
    #         claim: Claim
    #         evidence: Evidence 对象
    #
    #     Returns:
    #         "support", "refute", or "neutral"
    #     """
    #     # 临时设置：立场判断不需要搜索
    #     original_search = self.llm.enable_search
    #     original_force = self.llm.force_search
    #     self.llm.enable_search = False
    #     self.llm.force_search = False
    #
    #     try:
    #         result = self.stance_chain.invoke({
    #             "claim": claim,
    #             "evidence_content": evidence.content[:500],
    #             "evidence_source": evidence.source
    #         })
    #
    #         text = result.get('text', '').strip().lower()
    #
    #         # 解析输出格式: support/refute/neutral
    #         if 'support' in text:
    #             return 'support'
    #         elif 'refute' in text:
    #             return 'refute'
    #         elif 'neutral' in text:
    #             return 'neutral'
    #         else:
    #             # 默认中性
    #             return 'neutral'
    #
    #     except Exception as e:
    #         print(f"⚠ 立场判断failed: {e}")
    #         # 默认中性
    #         return 'neutral'
    #     finally:
    #         # 恢复原始配置
    #         self.llm.enable_search = original_search
    #         self.llm.force_search = original_force

    def make_verdict(
        self,
        claim: str,
        accepted_evidences: List,
        all_arguments: List,
        attack_relation: List,
        all_evidences_count: int
    ) -> Verdict:
        """
        生成最终Verdict

        Args:
            claim: Claim
            accepted_evidences: 被接受的证据列表
            all_evidences_count: 所有证据总数

        Returns:
            Verdict 对象
        """
        # if len(accepted_evidences)==0:
        #     return Verdict(
        #         decision="Not Enough Evidence",
        #         confidence=0.3,
        #         reasoning="没有被接受的证据，无法判断。",
        #         key_evidence_ids=[],
        #         accepted_evidence_ids=[],
        #         pro_strength=0.0,
        #         con_strength=0.0,
        #         total_evidences=all_evidences_count,
        #         accepted_evidences=0
        #     )



        # 4. 调用 LLM 生成Verdict
        # 启用完整搜索：这是最关键的决策，值得投入最多资源
        original_search = self.llm.enable_search
        original_force = self.llm.force_search
        original_strategy = self.llm.search_strategy

        self.llm.enable_search = True
        self.llm.force_search = True
        self.llm.search_strategy = "max"
        # self.llm.enable_thinking = True


        try:
            result = self.verdict_chain.invoke({
                "claim": claim,
                "accepted_arguments": accepted_evidences,
                "all_arguments": all_arguments,
                "attack_relation": attack_relation
            })

            text = result.get('text', '')

            # 解析输出
            decision = "Not Enough Evidence"
            reasoning = text
            confidence_from_llm = None

            # 首先尝试解析JSON格式
            import re
            json_match = re.search(r'\{[^{}]*"verdict"[^{}]*\}', text, re.DOTALL)
            if json_match:
                try:
                    import json as json_lib
                    json_str = json_match.group(0)
                    # 尝试修复可能的JSON格式问题
                    json_str = json_str.replace("'", '"')  # 单引号转双引号
                    parsed_json = json_lib.loads(json_str)

                    if "verdict" in parsed_json:
                        verdict_value = parsed_json["verdict"]
                        if verdict_value == "Supported":
                            decision = "Supported"
                        elif verdict_value == "Refuted":
                            decision = "Refuted"
                        elif verdict_value == "Not Enough Evidence":
                            decision = "Not Enough Evidence"

                    if "justification" in parsed_json:
                        reasoning = parsed_json["justification"]
                    elif "reasoning" in parsed_json:
                        reasoning = parsed_json["reasoning"]

                    if "confidence" in parsed_json:
                        confidence_from_llm = parsed_json["confidence"]
                except Exception as e:
                    print(f"  ⚠ JSON解析failed，尝试文本解析: {e}")

            # 如果JSON解析failed，尝试文本格式解析
            if decision == "Not Enough Evidence" and "Verdict:" in text:
                lines = text.split('\n')
                for line in lines:
                    if line.startswith("Verdict:"):
                        decision_text = line.replace("Verdict:", "").strip()
                        if "Supported" in decision_text:
                            decision = "Supported"
                        elif "Refuted" in decision_text:
                            decision = "Refuted"
                        elif "Not Enough Evidence" in decision_text:
                            decision = "Not Enough Evidence"
                    elif line.startswith("推理:"):
                        reasoning = line.replace("推理:", "").strip()

            # 如果还是没找到，尝试直接从文本中搜索关键词
            if decision == "Not Enough Evidence":
                text_upper = text.upper()
                if '"verdict": "Supported"' in text or '"verdict":"Supported"' in text or 'verdict": "Supported' in text:
                    decision = "Supported"
                elif '"verdict": "Refuted"' in text or '"verdict":"Refuted"' in text or 'verdict": "Refuted' in text:
                    decision = "Refuted"
                elif '"verdict": "Not Enough Evidence"' in text or '"verdict":"Not Enough Evidence"' in text:
                    decision = "Not Enough Evidence"
                elif 'Supported' in text_upper[:500] and 'verdict' in text_upper[:500]:
                    decision = "Supported"
                elif 'Refuted' in text_upper[:500] and 'verdict' in text_upper[:500]:
                    decision = "Refuted"


            return Verdict(
                decision=decision,
                confidence=0.5,
                reasoning=reasoning
            )

        except Exception as e:
            print(f"⚠ Verdict生成failed: {e}")
            return Verdict(
                decision="Not Enough Evidence",
                confidence=0.5,
                reasoning=f"Verdict生成failed: {e}"
            )
        finally:
            # 恢复原始配置
            self.llm.enable_search = original_search
            self.llm.force_search = original_force
            self.llm.search_strategy = original_strategy
