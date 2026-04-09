"""
Step 2: Argumentation Graph Construction阶段

功能：
1. Loading证据池和子 claims
2. 提取论点
3. 合并相似论点
4. 检测攻击关系
5. 构建论证图

支持并行Processing多个样本
"""

import json
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent))

import config
from llm.qwen_client import QwenClient
from core.claim_graph import ClaimGraph
from tools.claim_attack_detector import ClaimAttackDetector
from tools.claim_argument_extractor import ClaimBasedArgumentExtractor
from tools.argument_merger import ArgumentMerger
from utils.qwen_wrapper import QwenLLMWrapper
from utils.models import SubClaim
from utils.retry_utils import call_with_retry


def build_argumentation_graph(
    step1_result_path: Path,
    output_dir: Path = None,
    claim_id: str = None
) -> Dict:
    """
    为单个 claim 构建论证图

    Args:
        step1_result_path: Step 1 的结果文件路径
        output_dir: 输出目录
        claim_id: claim 的唯一标识符

    Returns:
        结果字典，包含论证图
    """
    print(f"\n{'='*80}")
    print(f"Step 2: Argumentation Graph Construction - {claim_id or 'Single Claim'}")
    print(f"{'='*80}\n")

    # Loading Step 1 结果
    with open(step1_result_path, "r", encoding="utf-8") as f:
        step1_data = json.load(f)

    claim = step1_data["claim"]
    claim_id = claim_id or step1_data.get("claim_id", "single")
    sub_claims_data = step1_data["sub_claims"]
    evidence_pool = step1_data["evidence_pool"]

    print(f"Claim: {claim}")
    print(f"子 Claims: {len(sub_claims_data)} 个")
    print(f"证据池: {len(evidence_pool)} 条证据\n")

    # 重建 SubClaim 对象
    sub_claims = [SubClaim(**sc) for sc in sub_claims_data]

    # 初始化组件
    llm_client = QwenClient(config.DASHSCOPE_API_KEY)
    claim_graph = ClaimGraph(claim)
    attack_detector = ClaimAttackDetector(llm_client, claim)

    extractor_llm = QwenLLMWrapper(qwen_client=llm_client, enable_search=False)
    argument_extractor = ClaimBasedArgumentExtractor(llm=extractor_llm)

    merger_llm = QwenLLMWrapper(qwen_client=llm_client, enable_search=False)
    argument_merger = ArgumentMerger(llm=merger_llm)

    # Step 1: 按 (search_query, agent, round) 分组证据
    print("[证据分组]")
    evidence_groups = {}
    for evidence in evidence_pool:
        key = (evidence["search_query"], evidence["retrieved_by"], evidence["round_num"])
        if key not in evidence_groups:
            evidence_groups[key] = []
        evidence_groups[key].append(evidence)

    print(f"✓ 分为 {len(evidence_groups)} 组\n")

    # Step 2: 提取论点
    print("[论点提取]")
    all_extracted_points = []

    for (search_query, agent, round_num), evidences in evidence_groups.items():
        try:
            print(f"  Processing [{agent}] Round{round_num}轮 查询「{search_query[:40]}...」: {len(evidences)} 条证据")

            # 转换证据格式为 search_results 格式
            search_results = []
            for ev in evidences:
                search_results.append({
                    "title": ev.get("title", ""),
                    "content": ev.get("content", ""),
                    "url": ev.get("url", ""),
                    "published_time": ev.get("published_time"),
                    "retrieved_time": ev.get("retrieved_time")
                })

            # 提取论点（带重试）
            points = call_with_retry(
                argument_extractor.extract_points,
                claim=claim,
                sub_claims=sub_claims,
                search_results=search_results,
                agent_type=agent,
                round_num=round_num,
                search_query=search_query,
                max_retries=3,
                initial_delay=2.0,
                backoff_factor=2.0
            )
            print(f"    → 提取了 {len(points)} 个论点")
            all_extracted_points.extend(points)
        except Exception as e:
            print(f"  ❌ 提取failed（重试后仍failed）[{agent}] {search_query}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✓ 总共提取了 {len(all_extracted_points)} 个论点\n")

    # Step 3: 合并相似论点（带重试）
    print("[论点合并]")
    try:
        merged_points = call_with_retry(
            argument_merger.merge_similar_points,
            all_extracted_points,
            similarity_threshold=0.7,
            max_retries=3,
            initial_delay=2.0
        )
    except Exception as e:
        print(f"⚠ 论点合并failed（重试后仍failed）: {e}，使用原始论点")
        merged_points = all_extracted_points

    # Step 4: 添加到图
    for point in merged_points:
        claim_graph.add_point_node(point)

    # Step 5: 按轮次检测攻击关系（带重试）
    print("\n[攻击检测]")
    max_round = max([p.round_num for p in merged_points]) if merged_points else 0

    for round_num in range(1, max_round + 1):
        print(f"  检测Round {round_num} 轮的攻击关系...")
        try:
            attacks = call_with_retry(
                attack_detector.detect_attacks_for_round,
                claim_graph,
                round_num,
                max_retries=3,
                initial_delay=2.0
            )
            claim_graph.add_attacks(attacks)
            print(f"    → 发现 {len(attacks)} 个攻击关系")
        except Exception as e:
            print(f"  ⚠ 攻击检测failed（重试后仍failed）: {e}")

    # Step 6: 计算 Grounded Extension
    print("\n[计算 Grounded Extension]")
    accepted_ids = claim_graph.compute_grounded_extension()
    print(f"✓ 被接受的论点: {len(accepted_ids)} 个\n")

    # 统计信息
    stats = claim_graph.get_statistics()
    print(f"[统计] Pro:{stats['pro_points']}, Con:{stats['con_points']}, 总计:{stats['total_points']}")
    print(f"        攻击边:{len(claim_graph.attack_edges)}, 被接受:{len(accepted_ids)}\n")

    # 构建结果
    result = {
        "claim": claim,
        "claim_id": claim_id,
        "timestamp": datetime.now().isoformat(),
        "sub_claims": sub_claims_data,
        "statistics": {
            "total_points": stats['total_points'],
            "pro_points": stats['pro_points'],
            "con_points": stats['con_points'],
            "total_attacks": len(claim_graph.attack_edges),
            "accepted_points": len(accepted_ids),
            "defeated_points": stats['total_points'] - len(accepted_ids),
            "avg_evidence_per_point": stats.get('avg_evidence_per_point', 0)
        },
        "claim_graph": claim_graph.to_dict(),
        "accepted_point_ids": list(accepted_ids),
        "ground_truth": step1_data.get("ground_truth")
    }

    # 保存结果
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存论证图
        with open(output_dir / f"{claim_id}_step2_graph.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        print(f"✓ 论证图已saved to: {output_dir / f'{claim_id}_step2_graph.json'}")

    return result


async def process_graphs_parallel(
    step1_dir: Path,
    output_dir: Path,
    max_parallel: int = 8,
    pattern: str = "*_step1_evidence.json"
) -> List[Dict]:
    """
    并行Processing多个 claims 的Argumentation Graph Construction

    Args:
        step1_dir: Step 1 结果目录
        output_dir: 输出目录
        max_parallel: 最大并行数
        pattern: 文件匹配模式

    Returns:
        结果列表
    """
    # 查找所有 Step 1 结果文件
    step1_files = sorted(list(step1_dir.glob(pattern)))
    print(f"\n找到 {len(step1_files)} 个 Step 1 结果文件\n")

    semaphore = asyncio.Semaphore(max_parallel)

    async def process_one(step1_file: Path):
        async with semaphore:
            # 提取 claim_id
            claim_id = step1_file.stem.replace("_step1_evidence", "")

            # 断点续传：检查输出文件是否already exists
            output_file = output_dir / f"{claim_id}_step2_graph.json"
            if output_file.exists():
                print(f"\n⏭️  Skip {claim_id}（already exists）")
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        result = json.load(f)
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
                build_argumentation_graph,
                step1_file,
                output_dir,
                claim_id
            )

            print(f"✓ {claim_id} Processingcompleted")
            return result

    # 创建所有任务
    tasks = [process_one(f) for f in step1_files]

    # 并行执行
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Processing异常
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            claim_id = step1_files[i].stem.replace("_step1_evidence", "")
            print(f"❌ {claim_id} Processingfailed: {result}")
            final_results.append({
                "claim_id": claim_id,
                "error": str(result)
            })
        else:
            final_results.append(result)

    return final_results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Step 2: Argumentation Graph Construction")
    parser.add_argument("--step1-file", type=str, help="单个 Step 1 结果文件路径")
    parser.add_argument("--step1-dir", type=str, default="output_pipeline/step1_evidence")
    parser.add_argument("--output", type=str, default="output_pipeline/step2_graphs", help="输出目录")
    parser.add_argument("--max-parallel", type=int, default=10, help="最大并行数")
    parser.add_argument("--pattern", type=str, default="*_step1_evidence.json", help="文件匹配模式")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.step1_file:
        # Processing单个文件
        step1_file = Path(args.step1_file)
        if not step1_file.exists():
            print(f"❌ 文件不存在: {step1_file}")
            return

        claim_id = step1_file.stem.replace("_step1_evidence", "")
        result = build_argumentation_graph(
            step1_result_path=step1_file,
            output_dir=output_dir,
            claim_id=claim_id
        )
        print(f"\n✓ 构建了包含 {result['statistics']['total_points']} 个论点的论证图")

    elif args.step1_dir:
        # 批量Processing
        step1_dir = Path(args.step1_dir)
        if not step1_dir.exists():
            print(f"❌ 目录不存在: {step1_dir}")
            return

        print(f"\n{'='*80}")
        print(f"批量Processing模式")
        print(f"{'='*80}")
        print(f"Step 1 目录: {step1_dir}")
        print(f"并行数: {args.max_parallel}")
        print(f"{'='*80}\n")

        # 并行Processing
        results = asyncio.run(process_graphs_parallel(
            step1_dir=step1_dir,
            output_dir=output_dir,
            max_parallel=args.max_parallel,
            pattern=args.pattern
        ))

        # 保存汇总
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_claims": len(results),
            "successful": len([r for r in results if "error" not in r]),
            "failed": len([r for r in results if "error" in r]),
            "results": results
        }

        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n{'='*80}")
        print(f"Processingcompleted")
        print(f"{'='*80}")
        print(f"successful: {summary['successful']}/{summary['total_claims']}")
        print(f"failed: {summary['failed']}/{summary['total_claims']}")
        print(f"结果已saved to: {output_dir}")
        print(f"{'='*80}\n")

    else:
        print("❌ 请指定 --step1-file 或 --step1-dir")
        print("使用 --help 查看帮助")


if __name__ == "__main__":
    main()
