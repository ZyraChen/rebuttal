"""
ClaimGraph - 论点节点和攻击关系（Claim-based架构）

核心特点：
- 节点：ClaimPoint对象（从多个证据中总结的论点）
- 一个论点可以由多个证据支持
- 论点是对子claim的回答
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Set, Optional
import json
from utils.models import ClaimPoint, ClaimPointAttackEdge


class ClaimGraph:
    """
    论点图（Claim-based架构）

    核心概念:
    - 节点: ClaimPoint对象（从证据中提取并合并的论点）
    - 边: ClaimPointAttackEdge对象（论点之间的矛盾关系）
    - 计算: Grounded Extension（可接受的论点集合）
    """

    def __init__(self, claim: str):
        self.claim = claim
        self.point_nodes: Dict[str, ClaimPoint] = {}  # 论点节点
        self.attack_edges: List[ClaimPointAttackEdge] = []  # 攻击边

    def add_point_node(self, point: ClaimPoint):
        """添加论点节点"""
        self.point_nodes[point.id] = point

    def add_point_nodes(self, points: List[ClaimPoint]):
        """批量添加论点节点"""
        for point in points:
            self.add_point_node(point)

    def add_attack(self, edge: ClaimPointAttackEdge):
        """
        添加攻击边（支持双向攻击）

        验证规则：
        - 权威性高的可以攻击权威性低的
        - 时效性高的可以攻击时效性低的
        - 允许双向攻击存在
        - 相同方向的攻击只添加一次，保留攻击原因
        """
        attacker = self.point_nodes.get(edge.from_point_id)
        target = self.point_nodes.get(edge.to_point_id)

        if not attacker or not target:
            print(f"⚠ 攻击边的节点不存在 {edge.from_point_id} -> {edge.to_point_id}")
            return

        # 验证双优先级规则
        attacker_auth = attacker.get_authority_priority()
        target_auth = target.get_authority_priority()
        attacker_time = attacker.get_timeliness_priority()
        target_time = target.get_timeliness_priority()
        
        # 至少在一个维度上攻击者要高于目标
        valid_attack = False
        if edge.attack_type == "authority" and attacker_auth > target_auth:
            valid_attack = True
        elif edge.attack_type == "timeliness" and attacker_time > target_time:
            valid_attack = True
        elif edge.attack_type == "both" and (attacker_auth > target_auth or attacker_time > target_time):
            valid_attack = True
        
        if not valid_attack:
            print(f"⚠ 攻击被拒绝,优先级不足: Auth({attacker_auth} vs {target_auth}), Time({attacker_time} vs {target_time})")
            return

        # 检查是否already exists相同方向的攻击边
        existing_edge = None
        for existing in self.attack_edges:
            if existing.from_point_id == edge.from_point_id and existing.to_point_id == edge.to_point_id:
                existing_edge = existing
                break
        
        if existing_edge:
            # already exists，更新攻击类型和强度（如果新的更强）
            if edge.strength > existing_edge.strength:
                existing_edge.strength = edge.strength
                existing_edge.attack_type = edge.attack_type
                existing_edge.rationale = edge.rationale
                print(f"⚙ 更新攻击: {edge.from_point_id} --[{edge.attack_type}]--> {edge.to_point_id}")
            else:
                print(f"⚙ Skip重复攻击: {edge.from_point_id} --> {edge.to_point_id}")
            return

        self.attack_edges.append(edge)
        print(f"✓ 添加攻击: {edge.from_point_id} --[{edge.attack_type}]--> {edge.to_point_id}")

    def add_attacks(self, edges: List[ClaimPointAttackEdge]):
        """批量添加攻击边"""
        for edge in edges:
            self.add_attack(edge)

    def get_attackers(self, target_id: str) -> List[str]:
        """获取攻击某个节点的所有攻击者ID"""
        return [e.from_point_id for e in self.attack_edges if e.to_point_id == target_id]

    def get_targets(self, attacker_id: str) -> List[str]:
        """获取某个节点攻击的所有目标ID"""
        return [e.to_point_id for e in self.attack_edges if e.from_point_id == attacker_id]

    def get_nodes_by_agent(self, agent: str) -> List[ClaimPoint]:
        """获取某方的所有论点节点"""
        return [p for p in self.point_nodes.values() if p.retrieved_by == agent]

    def get_node_by_id(self, node_id: str) -> Optional[ClaimPoint]:
        """根据ID获取节点"""
        return self.point_nodes.get(node_id)

    def compute_grounded_extension(self) -> Set[str]:
        """
        计算Grounded Extension - 可接受的论点集合

        算法:
        一个论点节点被接受,当且仅当:
        1. 没有攻击者, OR
        2. 所有攻击者都被击败(即攻击者本身被接受的论点攻击)
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

            for pid in self.point_nodes.keys():
                if pid in accepted or pid in defeated:
                    continue

                # 检查所有攻击者
                attackers = self.get_attackers(pid)

                if not attackers:
                    # 没有攻击者 -> 接受
                    accepted.add(pid)
                    changed = True
                else:
                    # 所有攻击者都被击败 -> 接受
                    if all(att_id in defeated for att_id in attackers):
                        accepted.add(pid)
                        changed = True
                    # 如果有任何攻击者被接受 -> 击败
                    elif any(att_id in accepted for att_id in attackers):
                        defeated.add(pid)
                        changed = True

        return accepted

    def to_dict(self) -> dict:
        """序列化为字典（用于保存和恢复）"""
        return {
            "claim": self.claim,
            "point_nodes": {
                pid: point.model_dump() for pid, point in self.point_nodes.items()
            },
            "attack_edges": [edge.model_dump() for edge in self.attack_edges]
        }

    def get_statistics(self) -> dict:
        """获取统计信息"""
        pro_points = [p for p in self.point_nodes.values() if p.retrieved_by == 'pro']
        con_points = [p for p in self.point_nodes.values() if p.retrieved_by == 'con']

        return {
            "total_points": len(self.point_nodes),
            "pro_points": len(pro_points),
            "con_points": len(con_points),
            "total_attacks": len(self.attack_edges),
            "avg_pro_priority": sum(p.get_priority() for p in pro_points) / len(pro_points) if pro_points else 0,
            "avg_con_priority": sum(p.get_priority() for p in con_points) / len(con_points) if con_points else 0,
            "avg_evidence_per_point": sum(len(p.supporting_evidence_ids) for p in self.point_nodes.values()) / len(self.point_nodes) if self.point_nodes else 0
        }

    def print_graph(self):
        """打印图结构（调试用）"""
        print(f"\n=== Claim Graph for: {self.claim} ===")
        print(f"Total Claim Points: {len(self.point_nodes)}")
        print(f"Total Attacks: {len(self.attack_edges)}")

        print("\nClaim Points:")
        for pid, point in self.point_nodes.items():
            evidence_count = len(point.supporting_evidence_ids)
            print(f"  [{pid}] ({point.retrieved_by}) P={point.get_priority():.2f} Evidence={evidence_count}:")
            print(f"    {point.point_text[:80]}...")

        print("\nAttacks:")
        for edge in self.attack_edges:
            print(f"  {edge.from_point_id} --({edge.strength:.2f})--> {edge.to_point_id}")
            print(f"    Reason: {edge.rationale[:100]}...")
