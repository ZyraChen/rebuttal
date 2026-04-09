"""
证据池 - 存储和管理所有证据
"""

from typing import List, Dict, Optional
from utils.models import Evidence


class EvidencePool:
    """
    证据池
    双方agent共享的证据存储
    """

    def __init__(self):
        self.evidences: Dict[str, Evidence] = {}

    def add_evidence(self, evidence: Evidence):
        """添加证据"""
        if evidence.id not in self.evidences:
            self.evidences[evidence.id] = evidence

    def add_batch(self, evidences: List[Evidence]):
        """批量添加证据"""
        for evidence in evidences:
            self.add_evidence(evidence)

    def get_by_id(self, evidence_id: str) -> Optional[Evidence]:
        """根据ID获取证据"""
        return self.evidences.get(evidence_id)

    def get_by_agent(self, agent: str, round_num: int = None) -> List[Evidence]:
        """
        获取某个agent检索的证据
        可选:只返回特定轮次的证据
        """
        result = [e for e in self.evidences.values() if e.retrieved_by == agent]
        if round_num is not None:
            result = [e for e in result if e.round_num == round_num]
        return result

    def get_by_round(self, round_num: int) -> List[Evidence]:
        """获取某轮的所有证据"""
        return [e for e in self.evidences.values() if e.round_num == round_num]

    def get_high_quality(self, min_score: float = 0.6) -> List[Evidence]:
        """获取高质量证据"""
        return [e for e in self.evidences.values() if e.quality_score >= min_score]

    def get_by_credibility(self, credibility: str) -> List[Evidence]:
        """根据可信度筛选"""
        return [e for e in self.evidences.values() if e.credibility == credibility]

    def get_all(self) -> List[Evidence]:
        """获取所有证据"""
        return list(self.evidences.values())

    def __len__(self) -> int:
        return len(self.evidences)

    def __repr__(self):
        return f"EvidencePool({len(self)} evidences)"

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total = len(self.evidences)
        if total == 0:
            return {"total": 0, "pro": 0, "con": 0, "high_quality": 0, "high_credibility": 0}

        pro_count = len(self.get_by_agent("pro"))
        con_count = len(self.get_by_agent("con"))
        high_quality = len(self.get_high_quality())
        high_cred = len(self.get_by_credibility("High"))

        return {
            "total": total,
            "pro": pro_count,
            "con": con_count,
            "high_quality": high_quality,
            "high_credibility": high_cred
        }

    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "evidences": [e.model_dump() for e in self.evidences.values()]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EvidencePool':
        """从字典反序列化"""
        pool = cls()
        for e_data in data.get("evidences", []):
            pool.add_evidence(Evidence(**e_data))
        return pool
