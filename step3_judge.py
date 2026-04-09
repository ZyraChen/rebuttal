"""
Step 3: Judge Verdict阶段

功能：
1. Loading论证图
2. 基于被接受的论点生成最终Verdict
3. 保存Verdict结果和完整日志
4. 实时统计和显示准确率

支持并行Processing多个样本
"""

import json
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

import config
from llm.qwen_client import QwenClient
from core.claim_graph import ClaimGraph
from chains import JudgeChain
from utils.qwen_wrapper import QwenLLMWrapper
from utils.models import ClaimPoint, ClaimPointAttackEdge
from utils.retry_utils import call_with_retry


class RealtimeStats:
    """实时统计类"""
    def __init__(self):
        self.total_processed = 0
        self.total_correct = 0
        self.total_with_gt = 0
        self.verdict_counts = defaultdict(int)
        self.correct_by_verdict = defaultdict(int)
        self.total_by_verdict = defaultdict(int)
        self.lock = asyncio.Lock()

    async def update(self, result: Dict):
        """更新统计信息"""
        async with self.lock:
            self.total_processed += 1

            if "error" not in result and result.get("ground_truth"):
                self.total_with_gt += 1
                predicted = result["verdict"]["decision"]
                ground_truth = result["ground_truth"]

                self.verdict_counts[predicted] += 1
                self.total_by_verdict[ground_truth] += 1

                if result.get("correct"):
                    self.total_correct += 1
                    self.correct_by_verdict[ground_truth] += 1
            else:
                print(result)

    def get_summary(self) -> str:
        """生成统计摘要"""
        if self.total_with_gt == 0:
            return f"已Processing: {self.total_processed} | 无ground truth数据"

        accuracy = self.total_correct / self.total_with_gt

        summary = [
            f"\n{'='*80}",
            f"实时统计 (已Processing {self.total_processed} 条)",
            f"{'='*80}",
            f"有标注数据: {self.total_with_gt} 条",
            f"正确预测: {self.total_correct} 条",
            f"总体准确率: {accuracy:.2%}",
            f"",
            f"预测分布:"
        ]

        for verdict in ["Supported", "Refuted", "Not Enough Evidence"]:
            count = self.verdict_counts.get(verdict, 0)
            if count > 0:
                summary.append(f"  {verdict}: {count} 条")

        summary.append(f"")
        summary.append(f"各类别准确率:")

        for verdict in ["Supported", "Refuted", "Not Enough Evidence"]:
            total = self.total_by_verdict.get(verdict, 0)
            if total > 0:
                correct = self.correct_by_verdict.get(verdict, 0)
                acc = correct / total
                summary.append(f"  {verdict}: {correct}/{total} = {acc:.2%}")

        summary.append(f"{'='*80}\n")

        return "\n".join(summary)


def make_verdict(
    step2_result_path: Path,
    output_dir: Path = None,
    claim_id: str = None
) -> Dict:
    """
    为单个 claim 生成Verdict

    Args:
        step2_result_path: Step 2 的结果文件路径
        output_dir: 输出目录
        claim_id: claim 的唯一标识符

    Returns:
        结果字典，包含Verdict
    """
    print(f"\n{'='*80}")
    print(f"Step 3: Judge Verdict - {claim_id or 'Single Claim'}")
    print(f"{'='*80}\n")

    # Loading Step 2 结果
    with open(step2_result_path, "r", encoding="utf-8") as f:
        step2_data = json.load(f)

    claim = step2_data["claim"]
    claim_id = claim_id or step2_data.get("claim_id", "single")
    claim_graph_data = step2_data["claim_graph"]
    accepted_point_ids = set(step2_data["accepted_point_ids"])
    ground_truth = step2_data.get("ground_truth")

    print(f"Claim: {claim}")
    print(f"总论点: {step2_data['statistics']['total_points']} 个")
    print(f"被接受的论点: {len(accepted_point_ids)} 个\n")

    all_arguments=step2_data["claim_graph"]["point_nodes"]
    attack_relation=step2_data["claim_graph"]["attack_edges"]

    # 重建 ClaimGraph
    claim_graph = ClaimGraph(claim)

    # 恢复论点节点
    for point_data in claim_graph_data["point_nodes"].values():
        point = ClaimPoint(**point_data)
        claim_graph.add_point_node(point)

    # 恢复攻击边
    for edge_data in claim_graph_data["attack_edges"]:
        edge = ClaimPointAttackEdge(**edge_data)
        claim_graph.add_attack(edge)

    # 初始化 Judge
    llm_client = QwenClient(config.DASHSCOPE_API_KEY)
    judge_llm = QwenLLMWrapper(qwen_client=llm_client, enable_search=True)
    judge_chain = JudgeChain(llm=judge_llm)

    # 获取被接受的论点
    accepted_points = [
        claim_graph.get_node_by_id(pid)
        for pid in accepted_point_ids
        if claim_graph.get_node_by_id(pid)
    ]

    print(f"[Judge Verdict]")
    print(f"基于 {len(accepted_points)} 个被接受的论点生成Verdict...\n")

    # 生成Verdict（带重试）
    try:
        verdict = call_with_retry(
            judge_chain.make_verdict,
            claim,
            accepted_points,
            all_arguments,
            attack_relation,
            len(claim_graph.point_nodes),
            max_retries=3,
            initial_delay=2.0,
            backoff_factor=2.0
        )

        decision_emoji = {"Supported": "✓", "Refuted": "✗", "Not Enough Evidence": "❓"}
        print(f"Verdict: {decision_emoji.get(verdict.decision, '')} {verdict.decision}")
        print(f"置信度: {verdict.confidence:.2%}")
        print(f"\n推理:")
        print("-" * 80)
        print(verdict.reasoning)
        print("-" * 80)
    except Exception as e:
        print(f"❌ Judge Verdictfailed（重试后仍failed）: {e}")
        import traceback
        traceback.print_exc()
        raise

    # 构建完整日志
    print("\n[构建完整日志]")
    complete_log = _build_complete_log(
        claim=claim,
        claim_graph=claim_graph,
        verdict=verdict,
        sub_claims=step2_data.get("sub_claims", []),
        ground_truth=ground_truth,
        accepted_point_ids=accepted_point_ids
    )

    # 评估结果
    is_correct = None
    if ground_truth:
        is_correct = (verdict.decision == ground_truth)
        print(f"\nGround Truth: {ground_truth}")
        print(f"预测: {verdict.decision}")
        print(f"结果: {'✓ 正确' if is_correct else '✗ error'}")

    # 构建结果
    result = {
        "claim": claim,
        "claim_id": claim_id,
        "timestamp": datetime.now().isoformat(),
        "verdict": verdict.model_dump(),
        "ground_truth": ground_truth,
        "correct": is_correct,
        "complete_log": complete_log
    }

    # 保存结果
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存Verdict
        with open(output_dir / f"{claim_id}_step3_verdict.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        # 保存完整日志
        with open(output_dir / f"{claim_id}_complete_log.json", "w", encoding="utf-8") as f:
            json.dump(complete_log, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n✓ Verdict已saved to: {output_dir / f'{claim_id}_step3_verdict.json'}")
        print(f"✓ 完整日志已saved to: {output_dir / f'{claim_id}_complete_log.json'}")

    return result


def _build_complete_log(
    claim: str,
    claim_graph: ClaimGraph,
    verdict,
    sub_claims: List[Dict],
    ground_truth: str = None,
    accepted_point_ids: set = None
) -> Dict:
    """构建完整的运行日志"""
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

    # 3. 被接受的论点
    if accepted_point_ids is None:
        accepted_point_ids = claim_graph.compute_grounded_extension()

    accepted_points = []
    for pid in accepted_point_ids:
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
    defeated_ids = set(claim_graph.point_nodes.keys()) - accepted_point_ids
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
        "accepted_evidences": verdict.accepted_points
    }

    # 6. 统计信息
    stats = claim_graph.get_statistics()

    # 7. 构建完整日志
    complete_log = {
        "claim": claim,
        "ground_truth": ground_truth,
        "timestamp": datetime.now().isoformat(),

        "sub_claims": sub_claims,

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
            "grounded_extension": list(accepted_point_ids)
        },

        "verdict": verdict_data,

        "evaluation": {
            "predicted": verdict.decision,
            "ground_truth": ground_truth,
            "correct": verdict.decision == ground_truth if ground_truth else None
        }
    }

    return complete_log


async def process_verdicts_parallel(
    step2_dir: Path,
    output_dir: Path,
    max_parallel: int = 8,
    pattern: str = "*_step2_graph.json"
) -> List[Dict]:
    """
    并行Processing多个 claims 的Verdict生成，实时显示准确率统计

    Args:
        step2_dir: Step 2 结果目录
        output_dir: 输出目录
        max_parallel: 最大并行数
        pattern: 文件匹配模式

    Returns:
        结果列表
    """
    # 查找所有 Step 2 结果文件
    step2_files = sorted(list(step2_dir.glob(pattern)))
    print(f"\n找到 {len(step2_files)} 个 Step 2 结果文件\n")

    semaphore = asyncio.Semaphore(max_parallel)
    stats = RealtimeStats()

    async def process_one(step2_file: Path):
        async with semaphore:
            # 提取 claim_id
            claim_id = step2_file.stem.replace("_step2_graph", "")

            # 断点续传：检查输出文件是否already exists
            output_file = output_dir / f"{claim_id}_step3_verdict.json"
            if output_file.exists():
                print(f"\n⏭️  Skip {claim_id}（already exists）")
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        result = json.load(f)
                    await stats.update(result)
                    print(stats.get_summary())
                    return result
                except Exception as e:
                    print(f"⚠ Reading已有文件failed: {e}，将重新Processing")

            print(f"\n{'#'*70}")
            print(f"StartProcessing {claim_id}")
            print(f"{'#'*70}")

            # 在线程池中运行同步函数
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                make_verdict,
                step2_file,
                output_dir,
                claim_id
            )

            # 更新统计并显示
            await stats.update(result)
            print(f"\n✓ {claim_id} Processingcompleted")
            print(stats.get_summary())

            return result

    # 创建所有任务
    tasks = [process_one(f) for f in step2_files]

    # 并行执行
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Processing异常
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            claim_id = step2_files[i].stem.replace("_step2_graph", "")
            print(f"❌ {claim_id} Processingfailed: {result}")
            final_results.append({
                "claim_id": claim_id,
                "error": str(result)
            })
        else:
            final_results.append(result)

    # 显示最终统计
    print("\n" + "="*80)
    print("最终统计结果")
    print("="*80)
    print(stats.get_summary())

    return final_results



def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Step 3: Judge Verdict")
    parser.add_argument("--step2-file", type=str, help="单个 Step 2 结果文件路径")
    parser.add_argument("--step2-dir", type=str, help="Step 2 结果目录（批量Processing）")
    parser.add_argument("--output", type=str, default="output_step3_122", help="输出目录")
    parser.add_argument("--max-parallel", type=int, default=10, help="最大并行数")
    parser.add_argument("--pattern", type=str, default="*_step2_graph.json", help="文件匹配模式")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.step2_file:
        # Processing单个文件
        step2_file = Path(args.step2_file)
        if not step2_file.exists():
            print(f"❌ 文件不存在: {step2_file}")
            return

        claim_id = step2_file.stem.replace("_step2_graph", "")
        result = make_verdict(
            step2_result_path=step2_file,
            output_dir=output_dir,
            claim_id=claim_id
        )
        print(f"\n✓ Verdict: {result['verdict']['decision']} (置信度: {result['verdict']['confidence']:.2%})")

    elif args.step2_dir:
        # 批量Processing
        step2_dir = Path(args.step2_dir)
        if not step2_dir.exists():
            print(f"❌ 目录不存在: {step2_dir}")
            return

        print(f"\n{'='*80}")
        print(f"批量Processing模式")
        print(f"{'='*80}")
        print(f"Step 2 目录: {step2_dir}")
        print(f"并行数: {args.max_parallel}")
        print(f"{'='*80}\n")

        # 并行Processing
        results = asyncio.run(process_verdicts_parallel(
            step2_dir=step2_dir,
            output_dir=output_dir,
            max_parallel=args.max_parallel,
            pattern=args.pattern
        ))


    # 计算准确率
        valid_results = [r for r in results if "error" not in r and r.get("correct") is not None]
        if valid_results:
            correct = sum(1 for r in valid_results if r["correct"])
            accuracy = correct / len(valid_results)

            print(f"\n{'='*80}")
            print(f"最终评估结果")
            print(f"{'='*80}")
            print(f"总数: {len(valid_results)}")
            print(f"正确: {correct}")
            print(f"准确率: {accuracy:.2%}")
            print(f"{'='*80}\n")

        # 保存汇总
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_claims": len(results),
            "successful": len([r for r in results if "error" not in r]),
            "failed": len([r for r in results if "error" in r]),
            "accuracy": accuracy if valid_results else None,
            "correct": correct if valid_results else None,
            "results": [
                {
                    "claim_id": r["claim_id"],
                    "claim": r.get("claim", ""),
                    "predicted": r["verdict"]["decision"] if "verdict" in r else None,
                    "ground_truth": r.get("ground_truth"),
                    "confidence": r["verdict"]["confidence"] if "verdict" in r else None,
                    "correct": r.get("correct"),
                    "error": r.get("error")
                }
                for r in results
            ]
        }

        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        print(f"结果已saved to: {output_dir}")

    else:
        print("❌ 请指定 --step2-file 或 --step2-dir")
        print("使用 --help 查看帮助")


if __name__ == "__main__":
    main()