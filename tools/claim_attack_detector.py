"""
ClaimAttackDetector - 论点攻击关系检测器

检测论点之间是否存在矛盾
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Tuple

# 从mad导入共享组件
from llm.qwen_client import QwenClient

# 从mad_v2导入模型
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.models import ClaimPoint, ClaimPointAttackEdge
from core.claim_graph import ClaimGraph


class ClaimAttackDetector:
    """Argument Attack Relation Detector"""

    def __init__(self, llm_client: QwenClient, claim: str):
        self.llm = llm_client
        self.claim = claim

    def detect_attacks_for_round(
        self,
        claim_graph: ClaimGraph,
        round_num: int
    ) -> List[ClaimPointAttackEdge]:
        """
        检测本轮新增论点与已有论点之间的攻击关系
        """
        new_edges = []

        # 获取本轮新增的论点
        new_points = [p for p in claim_graph.point_nodes.values() if p.round_num == round_num]

        # 获取本轮之前的所有已有论点
        existing_points = [p for p in claim_graph.point_nodes.values() if p.round_num < round_num]

        if not new_points or not existing_points:
            print(f"\n[论点攻击检测] Skip检测（新论点: {len(new_points)}个, 已有论点: {len(existing_points)}个）")
            return new_edges

        print(f"\n[论点攻击检测] 检测 {len(new_points)} 个新论点 vs {len(existing_points)} 个已有论点")

        # 批量检测所有新论点与已有论点的冲突
        all_conflicts = self._batch_detect_conflicts(new_points, existing_points)

        # 根据冲突结果创建攻击边（支持双向攻击）
        for new_point, existing_point, is_conflict, rationale in all_conflicts:
            if not is_conflict:
                continue

            # 获取权威性和时效性优先级
            new_auth = new_point.get_authority_priority()
            existing_auth = existing_point.get_authority_priority()
            new_time = new_point.get_timeliness_priority()
            existing_time = existing_point.get_timeliness_priority()

            # 检查是否产生攻击（双向）
            # 攻击规则：权威性高攻击权威性低，时效性高攻击时效性低
            
            # 新论点 -> 旧论点的攻击
            if new_auth > existing_auth or new_time > existing_time:
                # 确定攻击类型和强度
                attack_types = []
                strength = 0.0
                
                if new_auth > existing_auth:
                    attack_types.append("authority")
                    strength += (new_auth - existing_auth) * 0.3
                
                if new_time > existing_time:
                    attack_types.append("timeliness")
                    strength += (new_time - existing_time) * 0.2
                
                attack_type = "both" if len(attack_types) == 2 else attack_types[0]
                
                edge = ClaimPointAttackEdge(
                    from_point_id=new_point.id,
                    to_point_id=existing_point.id,
                    strength=max(0.1, strength),
                    rationale=rationale,
                    round_num=round_num,
                    attack_type=attack_type
                )
                new_edges.append(edge)
            
            # 旧论点 -> 新论点的攻击（双向）
            if existing_auth > new_auth or existing_time > new_time:
                # 确定攻击类型和强度
                attack_types = []
                strength = 0.0
                
                if existing_auth > new_auth:
                    attack_types.append("authority")
                    strength += (existing_auth - new_auth) * 0.3
                
                if existing_time > new_time:
                    attack_types.append("timeliness")
                    strength += (existing_time - new_time) * 0.2
                
                attack_type = "both" if len(attack_types) == 2 else attack_types[0]
                
                edge = ClaimPointAttackEdge(
                    from_point_id=existing_point.id,
                    to_point_id=new_point.id,
                    strength=max(0.1, strength),
                    rationale=rationale,
                    round_num=round_num,
                    attack_type=attack_type
                )
                new_edges.append(edge)

        print(f"✓ 检测completed，发现 {len(new_edges)} 个攻击关系")
        return new_edges

    def _batch_detect_conflicts(
        self,
        new_points: List[ClaimPoint],
        existing_points: List[ClaimPoint]
    ) -> List[Tuple[ClaimPoint, ClaimPoint, bool, str]]:
        """Batch detect argument conflicts (single LLM call)"""
        # 构建批量检测Prompt
        prompt = self._build_batch_conflict_prompt(new_points, existing_points)

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                enable_search=False,
                force_search=False
            )

            # 解析响应
            conflicts = self._parse_conflict_response(response, new_points, existing_points)
            return conflicts

        except Exception as e:
            print(f"⚠ 批量冲突检测failed: {e}")
            return []

    def _build_batch_conflict_prompt(
        self,
        new_points: List[ClaimPoint],
        existing_points: List[ClaimPoint]
    ) -> str:
        """Build batch conflict detection prompt"""
        prompt = f"""You are a fact-checking expert specializing in argumentation analysis. Your task is to identify attack relationships between arguments for constructing an abstract argumentation framework.

**Original Claim:**
{self.claim}

**Task:**
Analyze all argument pairs to identify conflicts that constitute attack relationships in the argumentation framework. Specifically, examine the following combinations:
1. Each new argument against all existing arguments
2. Each new argument against other new arguments

**New ARGUMENTS:**
"""
        for i, point in enumerate(new_points, 1):
            prompt += f"{i}. [{point.id}] ({point.retrieved_by})\n"
            prompt += f"   Sub-claim: {point.sub_claim_text}\n"
            prompt += f"   Point: {point.point_text}\n\n"

        prompt += "\n**Existing ARGUMENTS:**\n"
        for i, point in enumerate(existing_points, 1):
            prompt += f"{i}. [{point.id}] ({point.retrieved_by})\n"
            prompt += f"   Sub-claim: {point.sub_claim_text}\n"
            prompt += f"   Point: {point.point_text}\n\n"

        prompt += """
        
**Conflict Definition:**
Two arguments are in conflict if they exhibit any of the following:
- They reach opposite or contradictory conclusions about the claim
- They present contradictory evidence or facts
- They employ mutually exclusive reasoning or logical premises
- One argument directly refutes or undermines the other's foundation

**Output Format:**
For each contradicting pair, output one line:
CONFLICT: <A1_id> vs <A2_id> | <rationale> 
- "A1_id": identifier of the first argument
- "A2_id": identifier of the second argument
- "rationale": category of conflict (contradictory_conclusion, contradictory_evidence, logical_opposition, or direct_refutation)

If no contradictions, output:
NO_CONFLICTS

Now analyze:"""

        return prompt

    def _parse_conflict_response(
        self,
        response: str,
        new_points: List[ClaimPoint],
        existing_points: List[ClaimPoint]
    ) -> List[Tuple[ClaimPoint, ClaimPoint, bool, str]]:
        """Parse conflict detection response"""
        results = []

        # 创建ID到Point的映射
        new_point_map = {point.id: point for point in new_points}
        existing_point_map = {point.id: point for point in existing_points}

        # 检查是否没有冲突
        if "NO_CONFLICTS" in response or "no conflicts" in response.lower():
            for new_point in new_points:
                for existing_point in existing_points:
                    results.append((new_point, existing_point, False, ""))
            return results

        # 解析冲突行
        conflict_pairs = set()
        for line in response.split('\n'):
            line = line.strip()
            if not line.startswith('CONFLICT:'):
                continue

            try:
                parts = line.replace('CONFLICT:', '').split('|')
                if len(parts) < 2:
                    continue

                ids_part = parts[0].strip()
                rationale = parts[1].strip()

                id_parts = ids_part.split(' vs ')
                if len(id_parts) != 2:
                    continue

                new_id = id_parts[0].strip()
                existing_id = id_parts[1].strip()

                new_point = new_point_map.get(new_id)
                existing_point = existing_point_map.get(existing_id)

                if new_point and existing_point:
                    results.append((new_point, existing_point, True, rationale))
                    conflict_pairs.add((new_id, existing_id))

            except Exception as e:
                print(f"  ⚠ 解析冲突行failed: {line[:100]}... Error: {e}")
                continue

        # 对于未提及的配对，标记为无冲突
        for new_point in new_points:
            for existing_point in existing_points:
                if (new_point.id, existing_point.id) not in conflict_pairs:
                    results.append((new_point, existing_point, False, ""))

        return results
