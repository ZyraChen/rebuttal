"""
Claim-based Workflow

Core Process:
1. Round 1: Decompose claim into sub-claims → Generate queries → Search → Question evidence with sub-claims → Generate arguments → Merge arguments
2. Round 2+: Counter-attack opponent's arguments

Differences from GenSpark:
- Decompose claim first
- Question evidence with sub-claims
- Multiple evidence can be summarized into the same argument (via ArgumentMerger)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import uuid as uuid_lib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import config
from core.claim_graph import ClaimGraph
from llm.qwen_client import QwenClient
from tools.jina_search import JinaSearch
from tools.claim_attack_detector import ClaimAttackDetector
from tools.claim_argument_extractor import ClaimBasedArgumentExtractor
from tools.argument_merger import ArgumentMerger
from chains.claim_decomposer import ClaimDecomposer
from utils.models import Verdict

# Import shared chains from mad
from chains import ProQueryChain, ConQueryChain, JudgeChain
from utils.qwen_wrapper import QwenLLMWrapper


def run_claim_workflow(claim: str, max_rounds: int = 2, checkpoint_dir: Path = None, resume_from_checkpoint: bool = False) -> dict:
    """
    Run Claim-based workflow

    Args:
        claim: Claim to verify
        max_rounds: Maximum rounds (recommended 2 rounds, because of claim decomposition)
        checkpoint_dir: Checkpoint save directory
        resume_from_checkpoint: Whether to resume from checkpoint

    Returns:
        Result dictionary
    """
    print(f"\n{'='*80}")
    print(f"Claim-based Debate System (MAD v2)")
    print(f"{'='*80}\n")
    print(f"Claim: {claim}\n")

    # Initialize
    llm_client = QwenClient(config.DASHSCOPE_API_KEY)
    jina = JinaSearch(config.JINA_API_KEY)
    claim_graph = ClaimGraph(claim)
    attack_detector = ClaimAttackDetector(llm_client, claim)

    # Variables for checkpoint recovery
    start_round = 1
    all_queries = []
    sub_claims = None

    # Try to recover from checkpoint
    if resume_from_checkpoint and checkpoint_dir:
        checkpoint_file = checkpoint_dir / "checkpoint.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)

                # 恢复claim_graph
                if "claim_graph" in checkpoint_data:
                    graph_data = checkpoint_data["claim_graph"]
                    claim_graph = ClaimGraph(claim)
                    # 恢复论点节点
                    from utils.models import ClaimPoint
                    for point_data in graph_data.get("point_nodes", {}).values():
                        point = ClaimPoint(**point_data)
                        claim_graph.add_point_node(point)
                    # 恢复攻击边
                    from utils.models import ClaimPointAttackEdge
                    for edge_data in graph_data.get("attack_edges", []):
                        edge = ClaimPointAttackEdge(**edge_data)
                        claim_graph.add_attack(edge)

                # 恢复子claims
                if "sub_claims" in checkpoint_data:
                    from utils.models import SubClaim
                    sub_claims = [SubClaim(**sc) for sc in checkpoint_data["sub_claims"]]

                # 恢复查询记录
                all_queries = checkpoint_data.get("all_queries", [])
                start_round = checkpoint_data.get("current_round", 1) + 1

                print(f"✓ 从checkpoint恢复：已completedRound {checkpoint_data.get('current_round', 0)} 轮，将从Round {start_round} 轮继续\n")
            except Exception as e:
                print(f"⚠ Checkpoint恢复failed: {e}，将从头Start\n")
                start_round = 1
                all_queries = []
                sub_claims = None

    # 创建组件
    decomposer_llm = QwenLLMWrapper(qwen_client=llm_client, enable_search=False)
    claim_decomposer = ClaimDecomposer(llm=decomposer_llm)

    extractor_llm = QwenLLMWrapper(qwen_client=llm_client, enable_search=False)
    argument_extractor = ClaimBasedArgumentExtractor(llm=extractor_llm)

    merger_llm = QwenLLMWrapper(qwen_client=llm_client, enable_search=False)
    argument_merger = ArgumentMerger(llm=merger_llm)

    pro_llm = QwenLLMWrapper(qwen_client=llm_client, enable_search=True, search_strategy="turbo")
    pro_chain = ProQueryChain(llm=pro_llm)

    con_llm = QwenLLMWrapper(qwen_client=llm_client, enable_search=True, search_strategy="turbo")
    con_chain = ConQueryChain(llm=con_llm)

    judge_llm = QwenLLMWrapper(qwen_client=llm_client, enable_search=False)
    judge_chain = JudgeChain(llm=judge_llm)

    # 创建checkpoint保存函数
    def save_checkpoint(round_num: int = None, is_final: bool = False):
        """Save current progress checkpoint"""
        if checkpoint_dir is None:
            return

        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # 构建当前状态
            checkpoint_data = {
                "claim": claim,
                "max_rounds": max_rounds,
                "current_round": round_num if round_num else 0,
                "is_complete": is_final,
                "timestamp": datetime.now().isoformat(),
                "claim_graph": claim_graph.to_dict(),
                "sub_claims": [sc.model_dump() for sc in sub_claims] if sub_claims else [],
                "all_queries": all_queries,
                "statistics": claim_graph.get_statistics()
            }

            # 保存checkpoint
            checkpoint_file = checkpoint_dir / "checkpoint.json"
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2, default=str)

            if round_num:
                print(f"  💾 Checkpoint已保存（Round{round_num}轮）")
            elif is_final:
                print(f"  💾 最终结果已保存")
        except Exception as e:
            print(f"  ⚠ Checkpoint保存failed: {e}")

    # 多轮辩论
    for round_num in range(start_round, max_rounds + 1):
        print(f"\n{'='*70}")
        print(f"Round {round_num}/{max_rounds} 轮")
        print(f"{'='*70}\n")

        # Round 1: 分解claim
        if round_num == 1:
            print("[Claim分解]")
            try:
                sub_claims = claim_decomposer.decompose(claim)
                if not sub_claims:
                    print("⚠ Claim分解返回空列表，使用默认分解")
                    # 创建默认子claim
                    from utils.models import SubClaim
                    import uuid
                    sub_claims = [
                        SubClaim(
                            id=f"subclaim_default_{uuid.uuid4().hex[:8]}",
                            text=claim,
                            parent_claim=claim,
                            verification_type="Overall",
                            importance=1.0
                        )
                    ]
                print(f"✓ decomposed into {len(sub_claims)} sub-claims")
                for sc in sub_claims:
                    print(f"  - [{sc.verification_type}] {sc.text[:60]}... (importance:{sc.importance})")
                print()
            except Exception as e:
                print(f"❌ Claimdecomposition failed: {e}")
                import traceback
                traceback.print_exc()
                # 使用默认分解
                from utils.models import SubClaim
                import uuid
                sub_claims = [
                    SubClaim(
                        id=f"subclaim_default_{uuid.uuid4().hex[:8]}",
                        text=claim,
                        parent_claim=claim,
                        verification_type="Overall",
                        importance=1.0
                    )
                ]
                print(f"✓ 使用默认分解: 1sub-claims\n")

        # 检查sub_claims
        if not sub_claims:
            print("❌ error：sub_claims为空，无法提取论点")
            continue

        # 生成查询
        print("[查询生成]")
        con_points = claim_graph.get_nodes_by_agent("con")
        pro_queries = pro_chain.generate_queries(
            claim=claim, round_num=round_num,
            opponent_evidences=con_points, existing_queries=all_queries
        )

        pro_points = claim_graph.get_nodes_by_agent("pro")
        con_queries = con_chain.generate_queries(
            claim=claim, round_num=round_num,
            opponent_evidences=pro_points, existing_queries=all_queries
        )

        print(f"Pro: {pro_queries}")
        print(f"Con: {con_queries}")
        all_queries.extend(pro_queries + con_queries)

        # 并发搜索
        print("\n[搜索]")
        search_queries = [(q, "pro", round_num) for q in pro_queries] + [(q, "con", round_num) for q in con_queries]

        search_results_map = {}
        with ThreadPoolExecutor(max_workers=min(6, len(search_queries))) as executor:
            def search_query(query_info):
                query, agent, r = query_info
                try:
                    print(f"🔍 [{agent.upper()}] {query}")
                    return (query, agent, r, jina.search(query, top_k=6), None)
                except Exception as e:
                    print(f"⚠ 搜索failed: {e}")
                    return (query, agent, r, [], e)

            futures = {executor.submit(search_query, q): q for q in search_queries}
            for future in as_completed(futures):
                query, agent, r, results, error = future.result()
                if not error and results:
                    search_results_map[(query, agent, r)] = results

        print(f"✓ 搜索completed: {len(search_results_map)} 组结果\n")

        # 提取论点
        print("[论点提取]")
        print(f"  子Claim数量: {len(sub_claims) if sub_claims else 0}")
        print(f"  搜索结果组数: {len(search_results_map)}")
        all_extracted_points = []

        for (query, agent, r), results in search_results_map.items():
            try:
                print(f"  Processing [{agent}] {query}: {len(results)} 个搜索结果")
                points = argument_extractor.extract_points(
                    sub_claims=sub_claims,
                    search_results=results,
                    agent_type=agent,
                    round_num=r,
                    search_query=query
                )
                print(f"    → 提取了 {len(points)} 个论点")
                all_extracted_points.extend(points)
            except Exception as e:
                print(f"  ❌ 提取failed [{agent}] {query}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n✓ 总共提取了 {len(all_extracted_points)} 个论点\n")

        # 合并相似论点
        print("[论点合并]")
        merged_points = argument_merger.merge_similar_points(all_extracted_points, similarity_threshold=0.7)

        # 添加到图
        for point in merged_points:
            claim_graph.add_point_node(point)

        # 攻击检测
        print("\n[攻击检测]")
        attacks = attack_detector.detect_attacks_for_round(claim_graph, round_num)
        claim_graph.add_attacks(attacks)

        # 统计
        stats = claim_graph.get_statistics()
        print(f"\n[统计] Pro:{stats['pro_points']}, Con:{stats['con_points']}, 总计:{stats['total_points']}\n")

        # 每轮End后保存checkpoint
        save_checkpoint(round_num=round_num)

    # JudgeVerdict
    print(f"\n{'='*80}")
    print("JudgeVerdict")
    print(f"{'='*80}\n")

    try:
        accepted_ids = claim_graph.compute_grounded_extension()
        accepted_points = [claim_graph.get_node_by_id(pid) for pid in accepted_ids if claim_graph.get_node_by_id(pid)]

        print(f"被接受的论点: {len(accepted_points)} 个")

        verdict = judge_chain.make_verdict(claim, accepted_points, len(claim_graph.point_nodes))
    except Exception as e:
        print(f"⚠ JudgeVerdictfailed: {e}")
        # 保存checkpoint以便恢复
        save_checkpoint()
        raise

    # 打印报告
    _print_final_report(claim, claim_graph, verdict, sub_claims)

    # 构建完整日志
    complete_log = _build_complete_log(
        claim=claim,
        claim_graph=claim_graph,
        verdict=verdict,
        sub_claims=sub_claims,
        ground_truth=None  # 将在 main 中设置
    )

    # 保存最终checkpoint
    save_checkpoint(is_final=True)

    return {
        "claim": claim,
        "verdict": verdict.model_dump(),
        "claim_graph_stats": stats,
        "sub_claims": [sc.model_dump() for sc in sub_claims] if sub_claims else [],
        "total_points": len(claim_graph.point_nodes),
        "arg_graph_data": claim_graph.to_dict(),
        "complete_log": complete_log
    }


def _print_final_report(claim, claim_graph, verdict, sub_claims):
    """Print final report"""
    print(f"\n\n{'='*80}")
    print(" 最终Verdict (Claim-based架构)")
    print(f"{'='*80}\n")
    print(f"Claim: {claim}\n")

    if sub_claims:
        print("子Claims:")
        for sc in sub_claims:
            print(f"  - [{sc.verification_type}] {sc.text} (importance:{sc.importance})")
        print()

    decision_emoji = {"Supported": "✓", "Refuted": "✗", "Not Enough Evidence": "❓"}
    print(f"Verdict: {decision_emoji.get(verdict.decision, '')} {verdict.decision}")
    print(f"置信度: {verdict.confidence:.2%}")
    print(f"\n推理:")
    print("-" * 80)
    print(verdict.reasoning)

    stats = claim_graph.get_statistics()
    print(f"\n统计:")
    print(f"总论点: {stats['total_points']}")
    print(f"平均每论点证据数: {stats['avg_evidence_per_point']:.1f}")


def _build_complete_log(claim, claim_graph, verdict, sub_claims, ground_truth=None):
    """
    构建完整的运行日志

    Args:
        claim: Claim
        claim_graph: 论点图
        verdict: Verdict结果
        sub_claims: 子claims列表
        ground_truth: 数据集中的真实标签

    Returns:
        完整日志字典
    """
    # 1. 所有论点
    all_points = []
    for point in claim_graph.point_nodes.values():
        all_points.append({
            "id": point.id,
            "point_text": point.point_text,
            "sub_claim_text": point.sub_claim_text,
            "supporting_evidence_ids": point.supporting_evidence_ids,
            "supporting_evidence_snippets": point.supporting_evidence_snippets,
            "source_urls": point.source_urls,
            "source_domains": point.source_domains,
            "credibility": point.credibility,
            "retrieved_by": point.retrieved_by,
            "round_num": point.round_num,
            "quality_score": point.quality_score,
            "confidence": point.confidence,
            "priority": point.get_priority(),
            "timestamp": point.timestamp.isoformat() if hasattr(point.timestamp, 'isoformat') else str(point.timestamp)
        })

    # 2. 攻击关系
    attack_edges = []
    for edge in claim_graph.attack_edges:
        attacker = claim_graph.get_node_by_id(edge.from_point_id)
        target = claim_graph.get_node_by_id(edge.to_point_id)

        attack_edges.append({
            "from_point_id": edge.from_point_id,
            "from_agent": attacker.retrieved_by if attacker else "unknown",
            "from_priority": attacker.get_priority() if attacker else 0,
            "to_point_id": edge.to_point_id,
            "to_agent": target.retrieved_by if target else "unknown",
            "to_priority": target.get_priority() if target else 0,
            "strength": edge.strength,
            "rationale": edge.rationale,
            "round_num": edge.round_num
        })

    # 3. 被接受的论点（Grounded Extension）
    accepted_ids = claim_graph.compute_grounded_extension()
    accepted_points = []
    for pid in accepted_ids:
        point = claim_graph.get_node_by_id(pid)
        if point:
            accepted_points.append({
                "id": point.id,
                "agent": point.retrieved_by,
                "priority": point.get_priority(),
                "point_text": point.point_text,
                "evidence_count": len(point.supporting_evidence_ids)
            })

    # 4. 被击败的论点
    defeated_ids = set(claim_graph.point_nodes.keys()) - accepted_ids
    defeated_points = []
    for pid in defeated_ids:
        point = claim_graph.get_node_by_id(pid)
        if point:
            attackers = claim_graph.get_attackers(pid)
            defeated_points.append({
                "id": point.id,
                "agent": point.retrieved_by,
                "priority": point.get_priority(),
                "defeated_by": list(attackers)
            })

    # 5. Verdict结果
    verdict_data = {
        "decision": verdict.decision,
        "confidence": verdict.confidence,
        "reasoning": verdict.reasoning,
        "key_evidence_ids": verdict.key_evidence_ids if hasattr(verdict, 'key_evidence_ids') else [],
        "pro_strength": verdict.pro_strength,
        "con_strength": verdict.con_strength,
        # "total_evidences": verdict.total_evidences,
        "accepted_evidences": verdict.accepted_points
    }

    # 6. 统计信息
    stats = claim_graph.get_statistics()

    # 7. 构建完整日志
    complete_log = {
        "claim": claim,
        "ground_truth": ground_truth,
        "timestamp": datetime.now().isoformat(),

        "sub_claims": [sc.model_dump() for sc in sub_claims] if sub_claims else [],

        "statistics": {
            "total_points": stats['total_points'],
            "pro_points": stats['pro_points'],
            "con_points": stats['con_points'],
            "total_attacks": len(attack_edges),
            "accepted_points": len(accepted_points),
            "defeated_points": len(defeated_points),
            "avg_evidence_per_point": stats.get('avg_evidence_per_point', 0)
        },

        "claim_points": {
            "all_points": all_points,
            "accepted_points": accepted_points,
            "defeated_points": defeated_points
        },

        "argumentation": {
            "attack_edges": attack_edges,
            "grounded_extension": list(accepted_ids)
        },

        "verdict": verdict_data,

        "evaluation": {
            "predicted": verdict.decision,
            "ground_truth": ground_truth,
            "correct": verdict.decision == ground_truth if ground_truth else None
        }
    }

    return complete_log


if __name__ == "__main__":
    test_claim = "小威廉姆斯是史上获得大满贯最多的女子网球运动员"
    result = run_claim_workflow(test_claim, max_rounds=2)
    print(f"\nVerdict: {result['verdict']['decision']}")
