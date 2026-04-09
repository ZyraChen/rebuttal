"""
ArgumentGraph - 论断节点和攻击关系（GenSpark风格）

与ArgumentationGraph的区别：
- 节点：Argument对象（从文档中提取的论断），而不是Evidence对象（搜索结果原文）
- 更结构化：每个节点是一个具体的论断/声明，而不是整段文档
"""

from typing import List, Dict, Set, Optional
import json
from utils.models import Argument, ArgumentAttackEdge


class ArgumentGraph:
    """
    论断图（GenSpark风格）

    核心概念:
    - 节点: Argument对象（从文档中提取的具体论断）
    - 边: ArgumentAttackEdge对象（论断之间的矛盾关系）
    - 计算: Grounded Extension（可接受的论断集合）
    """

    def __init__(self, claim: str):
        self.claim = claim
        self.argument_nodes: Dict[str, Argument] = {}  # 论断节点
        self.attack_edges: List[ArgumentAttackEdge] = []  # 攻击边

    def add_argument_node(self, argument: Argument):
        """添加论断节点"""
        self.argument_nodes[argument.id] = argument

    def add_argument_nodes(self, arguments: List[Argument]):
        """批量添加论断节点"""
        for argument in arguments:
            self.add_argument_node(argument)

    def add_attack(self, edge: ArgumentAttackEdge):
        """
        添加攻击边

        验证:攻击者优先级 > 被攻击者优先级
        """
        attacker = self.argument_nodes.get(edge.from_argument_id)
        target = self.argument_nodes.get(edge.to_argument_id)

        if not attacker or not target:
            print(f"⚠ 攻击边的节点不存在 {edge.from_argument_id} -> {edge.to_argument_id}")
            return

        # 验证优先级规则
        if attacker.get_priority() <= target.get_priority():
            print(f"⚠ 攻击被拒绝,优先级不足: {attacker.get_priority():.2f} <= {target.get_priority():.2f}")
            return

        self.attack_edges.append(edge)
        print(f"✓ 添加攻击: {edge.from_argument_id} -> {edge.to_argument_id}")

    def add_attacks(self, edges: List[ArgumentAttackEdge]):
        """批量添加攻击边"""
        for edge in edges:
            self.add_attack(edge)

    def get_attackers(self, target_id: str) -> List[str]:
        """获取攻击某个节点的所有攻击者ID"""
        return [e.from_argument_id for e in self.attack_edges if e.to_argument_id == target_id]

    def get_targets(self, attacker_id: str) -> List[str]:
        """获取某个节点攻击的所有目标ID"""
        return [e.to_argument_id for e in self.attack_edges if e.from_argument_id == attacker_id]

    def get_nodes_by_agent(self, agent: str) -> List[Argument]:
        """获取某方的所有论断节点"""
        return [a for a in self.argument_nodes.values() if a.retrieved_by == agent]

    def get_node_by_id(self, node_id: str) -> Optional[Argument]:
        """根据ID获取节点"""
        return self.argument_nodes.get(node_id)

    def compute_grounded_extension(self) -> Set[str]:
        """
        计算Grounded Extension - 可接受的论断集合

        算法:
        一个论断节点被接受,当且仅当:
        1. 没有攻击者, OR
        2. 所有攻击者都被击败(即攻击者本身被接受的论断攻击)
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

            for aid in self.argument_nodes.keys():
                if aid in accepted or aid in defeated:
                    continue

                # 检查所有攻击者
                attackers = self.get_attackers(aid)

                if not attackers:
                    # 没有攻击者 -> 接受
                    accepted.add(aid)
                    changed = True
                else:
                    # 所有攻击者都被击败 -> 接受
                    if all(att_id in defeated for att_id in attackers):
                        accepted.add(aid)
                        changed = True
                    # 如果有任何攻击者被接受 -> 击败
                    elif any(att_id in accepted for att_id in attackers):
                        defeated.add(aid)
                        changed = True

        return accepted

    def to_dict(self) -> dict:
        """序列化为字典（用于保存和恢复）"""
        return {
            "claim": self.claim,
            "argument_nodes": {
                aid: arg.model_dump() for aid, arg in self.argument_nodes.items()
            },
            "attack_edges": [edge.model_dump() for edge in self.attack_edges]
        }

    def get_statistics(self) -> dict:
        """获取统计信息"""
        pro_args = [a for a in self.argument_nodes.values() if a.retrieved_by == 'pro']
        con_args = [a for a in self.argument_nodes.values() if a.retrieved_by == 'con']

        return {
            "total_arguments": len(self.argument_nodes),
            "pro_arguments": len(pro_args),
            "con_arguments": len(con_args),
            "total_attacks": len(self.attack_edges),
            "avg_pro_priority": sum(a.get_priority() for a in pro_args) / len(pro_args) if pro_args else 0,
            "avg_con_priority": sum(a.get_priority() for a in con_args) / len(con_args) if con_args else 0
        }

    def print_graph(self):
        """打印图结构（调试用）"""
        print(f"\n=== Argument Graph for: {self.claim} ===")
        print(f"Total Arguments: {len(self.argument_nodes)}")
        print(f"Total Attacks: {len(self.attack_edges)}")

        print("\nArguments:")
        for aid, arg in self.argument_nodes.items():
            print(f"  [{aid}] ({arg.retrieved_by}) P={arg.get_priority():.2f}: {arg.claim_text[:60]}...")

        print("\nAttacks:")
        for edge in self.attack_edges:
            print(f"  {edge.from_argument_id} --({edge.strength:.2f})--> {edge.to_argument_id}")
            print(f"    Reason: {edge.rationale[:100]}...")
