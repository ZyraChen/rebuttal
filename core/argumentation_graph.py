"""
论辩图 - 证据节点和攻击关系
"""

from typing import List, Dict, Set, Optional
import json
from utils.models import Evidence, AttackEdge


class ArgumentationGraph:
    """
    论辩图

    核心概念:
    - 节点: Evidence对象(每个搜索结果)
    - 边: AttackEdge对象(高优先级证据攻击低优先级证据)
    - 计算: Grounded Extension(可接受的证据集合)
    """

    def __init__(self, claim: str):
        self.claim = claim
        self.evidence_nodes: Dict[str, Evidence] = {}  # 证据节点
        self.attack_edges: List[AttackEdge] = []  # 攻击边

    def add_evidence_node(self, evidence: Evidence):
        """添加证据节点"""
        self.evidence_nodes[evidence.id] = evidence

    def add_evidence_nodes(self, evidences: List[Evidence]):
        """批量添加证据节点"""
        for evidence in evidences:
            self.add_evidence_node(evidence)

    def add_attack(self, edge: AttackEdge):
        """
        添加攻击边
        验证:攻击者优先级 > 被攻击者优先级
        """
        attacker = self.evidence_nodes.get(edge.from_evidence_id)
        target = self.evidence_nodes.get(edge.to_evidence_id)

        if not attacker or not target:
            print(f"⚠ 攻击边的节点不存在 {edge.from_evidence_id} -> {edge.to_evidence_id}")
            return

        # 验证优先级规则
        if attacker.get_priority() <= target.get_priority():
            print(f"⚠ 攻击被拒绝,优先级不足: {attacker.get_priority():.2f} <= {target.get_priority():.2f}")
            return

        self.attack_edges.append(edge)
        print(f"✓ 添加攻击: {edge.from_evidence_id} -> {edge.to_evidence_id}")

    def add_attacks(self, edges: List[AttackEdge]):
        """批量添加攻击边"""
        for edge in edges:
            self.add_attack(edge)

    def get_attackers(self, target_id: str) -> List[str]:
        """获取攻击某个节点的所有攻击者ID"""
        return [e.from_evidence_id for e in self.attack_edges if e.to_evidence_id == target_id]

    def get_targets(self, attacker_id: str) -> List[str]:
        """获取某个节点攻击的所有目标ID"""
        return [e.to_evidence_id for e in self.attack_edges if e.from_evidence_id == attacker_id]

    def get_nodes_by_agent(self, agent: str) -> List[Evidence]:
        """获取某方的所有证据节点"""
        return [e for e in self.evidence_nodes.values() if e.retrieved_by == agent]

    def get_node_by_id(self, node_id: str) -> Optional[Evidence]:
        """根据ID获取节点"""
        return self.evidence_nodes.get(node_id)

    def compute_grounded_extension(self) -> Set[str]:
        """
        计算Grounded Extension - 可接受的证据集合

        算法:
        一个证据节点被接受,当且仅当:
        1. 没有攻击者, OR
        2. 所有攻击者都被击败(即攻击者本身被接受的节点攻击)
        """
        accepted = set()
        defeated = set()

        # 迭代直到稳定
        changed = True
        max_iterations = 100
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            for eid in self.evidence_nodes.keys():
                if eid in accepted or eid in defeated:
                    continue

                # 检查所有攻击者
                attackers = self.get_attackers(eid)

                if not attackers:
                    # 没有攻击者,接受
                    accepted.add(eid)
                    changed = True
                else:
                    # 检查攻击者是否都被击败
                    all_defeated = all(a in defeated for a in attackers)
                    if all_defeated:
                        accepted.add(eid)
                        changed = True

                    # 检查是否被已接受的节点攻击
                    for attacker in attackers:
                        if attacker in accepted:
                            defeated.add(eid)
                            changed = True
                            break

        print(f"\n[Grounded Extension] 接受 {len(accepted)} 个节点, 击败 {len(defeated)} 个节点")
        return accepted

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        pro_nodes = self.get_nodes_by_agent("pro")
        con_nodes = self.get_nodes_by_agent("con")

        return {
            "total_evidences": len(self.evidence_nodes),
            "total_attacks": len(self.attack_edges),
            "pro_evidences": len(pro_nodes),
            "con_evidences": len(con_nodes),
            "avg_pro_priority": sum(e.get_priority() for e in pro_nodes) / max(len(pro_nodes), 1),
            "avg_con_priority": sum(e.get_priority() for e in con_nodes) / max(len(con_nodes), 1)
        }

    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "claim": self.claim,
            "evidence_nodes": [e.model_dump() for e in self.evidence_nodes.values()],
            "attack_edges": [e.model_dump() for e in self.attack_edges],
            "statistics": self.get_statistics()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ArgumentationGraph':
        """从字典反序列化"""
        graph = cls(data.get("claim", ""))
        for e_data in data.get("evidence_nodes", []):
            graph.add_evidence_node(Evidence(**e_data))
        for edge_data in data.get("attack_edges", []):
            graph.attack_edges.append(AttackEdge(**edge_data))
        return graph

    def save_to_file(self, filepath: str):
        """saved toJSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, default=str)
