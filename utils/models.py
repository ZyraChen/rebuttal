"""
数据模型 - Claim-based架构

核心理念：
- SubClaim：claim分解后的子claim
- ClaimPoint：从证据中提取的论点（多个证据可能支持同一个论点）
- ClaimPointAttackEdge：论点之间的攻击关系
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class SubClaim(BaseModel):
    """
    子Claim
    从原始claim分解得到的需要验证的子断言
    """
    id: str
    text: str  # 子claim文本
    parent_claim: str  # 原始claim
    verification_type: str  # 验证类型（如：事实核查、时间核查、数量核查等）
    importance: float = 1.0  # importance（1.0=核心断言，0.5=次要细节）

    class Config:
        frozen = False


class ClaimPoint(BaseModel):
    """
    论点节点（Claim-based架构的核心）

    一个论点是从一个或多个证据中，针对某个子claim提问后总结得出的
    """
    id: str
    point_text: str  # 论点文本（对子claim的回答/立场）
    sub_claim_id: str  # 关联的子claim ID
    sub_claim_text: str  # 关联的子claim文本
    supporting_evidence_ids: List[str] = []  # 支持该论点的证据ID列表
    supporting_evidence_snippets: List[str] = []  # 证据片段
    source_urls: List[str] = []  # 来源URL列表
    source_domains: List[str] = []  # 来源域名列表
    credibility: Literal["High", "Medium", "Low"]  # 综合可信度
    retrieved_by: Literal["pro", "con"]  # 谁检索的
    round_num: int  # Round几轮
    timestamp: datetime = Field(default_factory=datetime.now)
    quality_score: float = 0.0  # 质量分数(0-1)
    confidence: float = 0.0  # 论点的置信度(0-1)
    
    # 新增：双优先级系统
    authority_priority: Literal["High", "Medium", "Low"] = "Low"  # 权威性优先级（3级）
    timeliness_priority: int = 0  # 时效性优先级（0=无时间，1-3=时间分级）
    evidence_published_times: List[Optional[str]] = []  # 支持证据的发布时间列表

    class Config:
        frozen = False

    def get_priority(self) -> float:
        """
        计算优先级分数（向后兼容的综合分数，用于排序）
        注意：攻击关系现在使用 get_authority_priority() 和 get_timeliness_priority()
        """
        cred_map = {"High": 1.0, "Medium": 0.6, "Low": 0.3}
        base_cred = cred_map.get(self.credibility, 0.5)

        # 综合考虑可信度、质量、置信度和证据数量
        evidence_score = min(1.0, len(self.supporting_evidence_ids) / 3.0)  # 越多证据越好
        priority = (base_cred * 0.4 + self.quality_score * 0.3 +
                   self.confidence * 0.2 + evidence_score * 0.1)

        return min(1.0, priority)
    
    def get_authority_priority(self) -> int:
        """
        获取权威性优先级（3级）
        返回: 3=High, 2=Medium, 1=Low
        """
        authority_map = {"High": 3, "Medium": 2, "Low": 1}
        return authority_map.get(self.authority_priority, 1)
    
    def get_timeliness_priority(self) -> int:
        """
        获取时效性优先级
        返回: 0=无时间信息, 1-3=按时间分级（越新越高）
        """
        return self.timeliness_priority

    # 兼容性属性（用于Judge）
    @property
    def content(self) -> str:
        """Compatible with Evidence.content field"""
        evidence_text = "\n".join([f"- {snippet}" for snippet in self.supporting_evidence_snippets[:3]])
        return f"论点: {self.point_text}\n\n支持证据:\n{evidence_text}"

    @property
    def source(self) -> str:
        """Compatible with Evidence.source field"""
        return ", ".join(self.source_domains[:3]) if self.source_domains else "未知"


class ClaimPointAttackEdge(BaseModel):
    """
    论点攻击边
    表示一个论点攻击另一个论点
    """
    from_point_id: str  # 攻击者论点ID
    to_point_id: str  # 被攻击者论点ID
    strength: float  # 攻击强度
    rationale: str  # 攻击理由（为什么这两个论点矛盾）
    round_num: int  # 哪一轮产生的攻击
    attack_type: Literal["authority", "timeliness", "both"] = "both"  # 攻击类型

    class Config:
        frozen = False


class Verdict(BaseModel):
    """Final Verdict"""
    decision: Literal["Supported", "Refuted", "Not Enough Evidence"]
    confidence: float  # 0-1
    reasoning: str  # 推理过程
    key_point_ids: List[str] = []  # 关键论点ID列表
    accepted_point_ids: List[str] = []  # 被接受的论点ID列表
    pro_strength: float = 0.0
    con_strength: float = 0.0
    total_points: int = 0
    accepted_points: int = 0


class ClaimData(BaseModel):
    """A claim in the dataset"""
    claim: str
    verdict: Optional[str] = None
    error_type: Optional[str] = None
    category: Optional[str] = None
    justification: Optional[str] = None
    evidence_sources: Optional[List[dict]] = None
    correct_answer: Optional[str] = None
    topic: Optional[str] = None
