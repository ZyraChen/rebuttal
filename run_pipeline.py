"""
统一运行脚本 - 运行所有三个步骤

功能：
1. Step 1: 证据收集
2. Step 2: 论证图构建
3. Step 3: Judge 判决

支持并行处理和灵活配置
"""

import os
os.environ["JINA_API_KEY"] = "jina_518b9cb292b249139bedce5123349109HnqXMjmaY94laLNX3J50eXfmd9E5"

import json
import sys
import asyncio
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from step1_evidence_collection import process_claims_parallel as step1_process
from step2_argumentation_graph import process_graphs_parallel as step2_process
from step3_judge import process_verdicts_parallel as step3_process


def _format_time(seconds: float) -> str:
    """将秒数格式化为易读字符串"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds/60:.1f}min ({seconds:.0f}s)"


def _print_timing_summary(step_timings: dict):
    """打印各步骤的计时汇总表"""
    print(f"\n{'='*65}")
    print(f"  计时汇总")
    print(f"{'='*65}")
    print(f"  {'步骤':<20} {'总用时':>10}  {'成功样本':>8}  {'平均/样本':>12}")
    print(f"  {'-'*61}")

    total_wall = 0.0
    total_success = 0

    for step_name, info in step_timings.items():
        wall   = info["wall_time"]
        n      = info["n_success"]
        avg    = wall / n if n > 0 else 0.0
        total_wall += wall
        total_success = max(total_success, n)  # 各步骤处理同批数据
        print(f"  {step_name:<20} {_format_time(wall):>10}  {n:>8}  {_format_time(avg):>12}")

    print(f"  {'-'*61}")
    # 全流程平均：三步骤总时间 / 样本数（注意并行时墙钟时间 < 串行累计）
    if total_success > 0:
        print(f"  {'全流程合计':<20} {_format_time(total_wall):>10}  {total_success:>8}  "
              f"{_format_time(total_wall / total_success):>12}")
    print(f"{'='*65}\n")
    if total_success > 1:
        print(f"  注：各步骤内部为并行执行（max_parallel 个 claim 同时跑），")
        print(f"      「平均/样本」= 步骤墙钟时间 ÷ 成功样本数，反映吞吐效率。\n")


async def run_all_steps(
    dataset_path: str,
    output_base_dir: str = "output_pipeline",
    max_samples: int = None,
    max_rounds: int = 3,
    max_parallel: int = 8,
    start_index: int = 0,
    steps_to_run: list = [1, 2, 3]
):
    """
    运行完整的三步骤流程

    Args:
        dataset_path: 数据集路径
        output_base_dir: 输出基础目录
        max_samples: 最大处理样本数
        max_rounds: 搜索轮次
        max_parallel: 最大并行数
        start_index: 起始索引
        steps_to_run: 要运行的步骤列表 [1, 2, 3]
    """
    print(f"\n{'='*80}")
    print(f"ArgCheck Pipeline - 三步骤流程")
    print(f"{'='*80}")
    print(f"数据集: {dataset_path}")
    print(f"输出目录: {output_base_dir}")
    print(f"最大样本数: {max_samples or '全部'}")
    print(f"搜索轮次: {max_rounds}")
    print(f"并行数: {max_parallel}")
    print(f"要运行的步骤: {steps_to_run}")
    print(f"{'='*80}\n")

    # 创建输出目录
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    step1_dir = output_base / "step1_evidence"
    step2_dir = output_base / "step2_graphs"
    step3_dir = output_base / "step3_verdicts"

    overall_start = time.time()

    # 用于汇总计时的字典
    step_timings = {}

    # ── Step 1: 证据收集 ──────────────────────────────────────────────────
    if 1 in steps_to_run:
        print(f"\n{'='*80}")
        print(f"Step 1: 证据收集")
        print(f"{'='*80}\n")

        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        if max_samples:
            dataset = dataset[:max_samples]
        print(f"加载了 {len(dataset)} 个样本\n")

        step1_start = time.time()
        step1_results = await step1_process(
            claims=dataset,
            max_rounds=max_rounds,
            output_dir=step1_dir,
            max_parallel=max_parallel,
            start_index=start_index
        )
        step1_wall = time.time() - step1_start

        n1_success = len([r for r in step1_results if "error" not in r])
        step_timings["Step 1 (证据收集)"] = {
            "wall_time": step1_wall,
            "n_success": n1_success,
        }

        print(f"\n✓ Step 1 完成")
        print(f"  总用时   : {_format_time(step1_wall)}")
        print(f"  成功样本 : {n1_success}/{len(step1_results)}")
        if n1_success > 0:
            print(f"  平均/样本: {_format_time(step1_wall / n1_success)}")
    else:
        print(f"\n⏭️  跳过 Step 1")

    # ── Step 2: 论证图构建 ────────────────────────────────────────────────
    if 2 in steps_to_run:
        print(f"\n{'='*80}")
        print(f"Step 2: 论证图构建")
        print(f"{'='*80}\n")

        step2_start = time.time()
        step2_results = await step2_process(
            step1_dir=step1_dir,
            output_dir=step2_dir,
            max_parallel=max_parallel
        )
        step2_wall = time.time() - step2_start

        n2_success = len([r for r in step2_results if "error" not in r])
        step_timings["Step 2 (论证图)"] = {
            "wall_time": step2_wall,
            "n_success": n2_success,
        }

        print(f"\n✓ Step 2 完成")
        print(f"  总用时   : {_format_time(step2_wall)}")
        print(f"  成功样本 : {n2_success}/{len(step2_results)}")
        if n2_success > 0:
            print(f"  平均/样本: {_format_time(step2_wall / n2_success)}")
    else:
        print(f"\n⏭️  跳过 Step 2")

    # ── Step 3: Judge 判决 ────────────────────────────────────────────────
    if 3 in steps_to_run:
        print(f"\n{'='*80}")
        print(f"Step 3: Judge 判决")
        print(f"{'='*80}\n")

        step3_start = time.time()
        step3_results = await step3_process(
            step2_dir=step2_dir,
            output_dir=step3_dir,
            max_parallel=max_parallel
        )
        step3_wall = time.time() - step3_start

        n3_success = len([r for r in step3_results if "error" not in r])
        step_timings["Step 3 (Judge)"] = {
            "wall_time": step3_wall,
            "n_success": n3_success,
        }

        print(f"\n✓ Step 3 完成")
        print(f"  总用时   : {_format_time(step3_wall)}")
        print(f"  成功样本 : {n3_success}/{len(step3_results)}")
        if n3_success > 0:
            print(f"  平均/样本: {_format_time(step3_wall / n3_success)}")

        # 准确率
        valid_results = [r for r in step3_results
                         if "error" not in r and r.get("correct") is not None]
        if valid_results:
            correct = sum(1 for r in valid_results if r["correct"])
            accuracy = correct / len(valid_results)
            print(f"\n{'='*80}")
            print(f"最终评估结果")
            print(f"{'='*80}")
            print(f"总数: {len(valid_results)}  正确: {correct}  准确率: {accuracy:.2%}")
            print(f"{'='*80}\n")
    else:
        print(f"\n⏭️  跳过 Step 3")

    # ── 汇总计时 ──────────────────────────────────────────────────────────
    overall_time = time.time() - overall_start
    _print_timing_summary(step_timings)

    print(f"总用时: {_format_time(overall_time)}  |  输出目录: {output_base}\n")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="ArgCheck Pipeline - 运行所有步骤")
    parser.add_argument("--dataset", type=str, required=True, default="data/dataset_latest.json")
    parser.add_argument("--output", type=str, default="output_pipeline", help="输出基础目录")
    parser.add_argument("--max-samples", type=int, default=None, help="最大处理样本数")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-parallel", type=int, default=1, help="最大并行数")
    parser.add_argument("--start-index", type=int, default=0, help="起始索引")
    parser.add_argument("--steps", type=str, default="2,3",
                        help="要运行的步骤，用逗号分隔，如 '1,2,3' 或 '2,3'")

    args = parser.parse_args()

    steps_to_run = [int(s.strip()) for s in args.steps.split(",")]

    asyncio.run(run_all_steps(
        dataset_path=args.dataset,
        output_base_dir=args.output,
        max_samples=args.max_samples,
        max_rounds=args.max_rounds,
        max_parallel=args.max_parallel,
        start_index=args.start_index,
        steps_to_run=steps_to_run
    ))


if __name__ == "__main__":
    main()
