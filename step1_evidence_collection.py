"""
Step 1: Evidence Collection Phase

Features:
1. Decompose claim into sub-claims
2. Multi-round adversarial search to build evidence pool
3. Save evidence pool (including timestamps) and sub-claims

Support parallel processing of multiple samples
"""

import json
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import config
from llm.qwen_client import QwenClient
from tools.jina_search import JinaSearch
from chains.claim_decomposer import ClaimDecomposer
from chains import ProQueryChain, ConQueryChain
from utils.qwen_wrapper import QwenLLMWrapper
from utils.models import SubClaim
from utils.retry_utils import call_with_retry, call_with_retry_until_success
from tools.evidence_filter import EvidenceFilter

def collect_evidence_for_claim(
    claim: str,
    max_rounds: int = 3,
    output_dir: Path = None,
    claim_id: str = None
) -> Dict:
    """
    为单个 claim 收集证据

    Args:
        claim: 要核查的 claim
        max_rounds: 最大搜索轮次
        output_dir: 输出目录
        claim_id: claim 的唯一标识符

    Returns:
        结果字典，包含子 claims 和证据池
    """
    print(f"\n{'='*80}")
    print(f"Step 1: Evidence Collection - {claim_id or 'Single Claim'}")
    print(f"{'='*80}\n")
    print(f"Claim: {claim}\n")

    # Initialize components
    llm_client = QwenClient(config.DASHSCOPE_API_KEY)
    jina = JinaSearch(config.JINA_API_KEY)

    # Create chains
    decomposer_llm = QwenLLMWrapper(qwen_client=llm_client, enable_search=False)
    claim_decomposer = ClaimDecomposer(llm=decomposer_llm)

    pro_llm = QwenLLMWrapper(qwen_client=llm_client, enable_search=True, search_strategy="turbo")
    pro_chain = ProQueryChain(llm=pro_llm)

    con_llm = QwenLLMWrapper(qwen_client=llm_client, enable_search=True, search_strategy="turbo")
    con_chain = ConQueryChain(llm=con_llm)

    # Step 1: Decompose claim (with retry, max 10 times)
    print("[Claim 分解]")
    try:
        sub_claims = call_with_retry_until_success(
            claim_decomposer.decompose,
            claim,
            validate_result=lambda x: x and len(x) > 0,  # 验证返回非空列表
            max_retries=10,  # 最多重试10次，failed则Skip
            initial_delay=2.0,
            backoff_factor=1.5,
            max_delay=30.0
        )
        
        # 检查是否返回了有效结果
        if not sub_claims or len(sub_claims) == 0:
            print(f"❌ Claimdecomposition failed（10retries still returned empty list）")
            raise ValueError("Claim decomposition failed after 10 retries")
        
        print(f"✓ decomposed into {len(sub_claims)} sub-claims")
        for sc in sub_claims:
            print(f"  - [{sc.verification_type}] {sc.text[:60]}... (importance:{sc.importance})")
        print()
    except Exception as e:
        print(f"❌ Claimdecomposition failed: {e}")
        print(f"⏭️  Skip this data, continue with next")
        # 抛出异常，让外层捕获并Skip
        raise RuntimeError(f"Claim decomposition failed: {e}")

    # Step 2: 多round search收集证据
    all_queries = []  # 存储格式: [{"query": str, "agent": str, "round_num": int}, ...]
    evidence_pool = []  # 证据池
    evidence_id_counter = 0

    # 用于模拟的临时图（仅用于查询生成）
    pro_points_cache = []
    con_points_cache = []

    for round_num in range(1, max_rounds + 1):
        print(f"\n{'='*70}")
        print(f"Round {round_num}/{max_rounds} round search")
        print(f"{'='*70}\n")

        # Generate queries (max 10 retries)
        print("[Query Generation]")
        # Extract pure query string list for generate_queries to use
        existing_query_strings = [q["query"] if isinstance(q, dict) else q for q in all_queries]
        
        print("生成 Pro 查询...")
        try:
            pro_queries = call_with_retry_until_success(
                pro_chain.generate_queries,
                claim=claim,
                round_num=round_num,
                opponent_evidences=con_points_cache,
                existing_queries=existing_query_strings,
                validate_result=lambda x: x and len(x) > 0,  # 验证返回非空列表
                max_retries=10,  # 最多重试10次
                initial_delay=2.0,
                backoff_factor=1.5,
                max_delay=30.0
            )
            
            if not pro_queries or len(pro_queries) == 0:
                print(f"❌ Pro query generation failed（10retries still returned empty list）")
                raise ValueError("Pro query generation failed after 10 retries")
            
            print(f"✓ Pro query generation successful, total {len(pro_queries)} queries")
        except Exception as e:
            print(f"❌ Pro query generation failed: {e}")
            print(f"⏭️  Skip this data, continue with next")
            raise RuntimeError(f"Pro query generation failed: {e}")

        print("生成 Con 查询...")
        try:
            con_queries = call_with_retry_until_success(
                con_chain.generate_queries,
                claim=claim,
                round_num=round_num,
                opponent_evidences=pro_points_cache,
                existing_queries=existing_query_strings,
                validate_result=lambda x: x and len(x) > 0,  # 验证返回非空列表
                max_retries=10,  # 最多重试10次
                initial_delay=2.0,
                backoff_factor=1.5,
                max_delay=30.0
            )
            
            if not con_queries or len(con_queries) == 0:
                print(f"❌ Con query generation failed（10retries still returned empty list）")
                raise ValueError("Con query generation failed after 10 retries")
            
            print(f"✓ Con query generation successful, total {len(con_queries)} queries")
        except Exception as e:
            print(f"❌ Con query generation failed: {e}")
            print(f"⏭️  Skip this data, continue with next")
            raise RuntimeError(f"Con query generation failed: {e}")

        print(f"Pro queries: {pro_queries}")
        print(f"Con queries: {con_queries}")

        # Record queries (including round number and agent info)

        for q in pro_queries:
            all_queries.append({"query": q, "agent": "pro", "round_num": round_num})

        for q in con_queries:
            all_queries.append({"query": q, "agent": "con", "round_num": round_num})

        # Concurrent search
        print("\n[Search Evidence]")
        search_queries = [(q, "pro", round_num) for q in pro_queries] + \
                        [(q, "con", round_num) for q in con_queries]

        with ThreadPoolExecutor(max_workers=min(6, len(search_queries))) as executor:
            def search_query(query_info):
                query, agent, r = query_info
                try:
                    print(f"🔍 [{agent.upper()}] {query}")
                    # Use retry for search
                    results = call_with_retry(
                        jina.search,
                        query,
                        top_k=config.MAX_SEARCH_RESULTS_PER_QUERY,
                        max_retries=40,
                        initial_delay=2.0,
                        backoff_factor=2.0
                    )
                    return (query, agent, r, results, None)
                except Exception as e:
                    print(f"⚠ 搜索failed（重试后仍failed）: {e}")
                    return (query, agent, r, [], e)

            futures = {executor.submit(search_query, q): q for q in search_queries}
            for future in as_completed(futures):
                query, agent, r, results, error = future.result()
                if not error and results:
                    # Add search results to evidence pool
                    for result in results:
                        evidence_id = f"evidence_{claim_id or 'single'}_{evidence_id_counter:04d}"
                        evidence_id_counter += 1

                        evidence_entry = {
                            "id": evidence_id,
                            "title": result.get("title", ""),
                            "content": result.get("content", ""),
                            "url": result.get("url", ""),
                            "published_time": result.get("published_time"),
                            "retrieved_time": result.get("retrieved_time", datetime.now().isoformat()),
                            "search_query": query,
                            "retrieved_by": agent,
                            "round_num": r
                        }
                        evidence_pool.append(evidence_entry)

                        # Update cache (for next round query generation)
                        # Create a simple object with source and content attributes
                        class SimpleEvidence:
                            def __init__(self, content, source):
                                self.content = content
                                self.source = source

                        cache_entry = SimpleEvidence(
                            content=result.get("content", ""),
                            source=result.get("url", "unknown")
                        )
                        if agent == "pro":
                            pro_points_cache.append(cache_entry)
                        else:
                            con_points_cache.append(cache_entry)

            print(f"✓ Round {round_num} 轮收集了 {len([e for e in evidence_pool if e['round_num'] == round_num])} 条证据")

        print(f"\n✓ Total collected {len(evidence_pool)} pieces of evidence")

        # Step 3: Filter irrelevant and duplicate evidence

        if evidence_pool:

            filter_llm = QwenLLMWrapper(qwen_client=llm_client, enable_search=False)

            evidence_filter = EvidenceFilter(llm=filter_llm)

            try:

                evidence_pool = call_with_retry(
                    evidence_filter.filter_evidence,
                    claim=claim,
                    evidence_pool=evidence_pool,
                    batch_size=10,
                    max_retries=2,
                    initial_delay=2.0
                )

            except Exception as e:

                print(f"⚠ Evidence filtering failed: {e}, keep all evidence")

        print(f"✓ Final evidence count: {len(evidence_pool)}\n")

    # Build result
    result = {
        "claim": claim,
        "claim_id": claim_id,
        "timestamp": datetime.now().isoformat(),
        "sub_claims": [sc.model_dump() for sc in sub_claims],
        "evidence_pool": evidence_pool,
        "search_queries": all_queries,
        "statistics": {
            "total_evidence": len(evidence_pool),
            "total_queries": len(all_queries),
            "pro_evidence": len([e for e in evidence_pool if e["retrieved_by"] == "pro"]),
            "con_evidence": len([e for e in evidence_pool if e["retrieved_by"] == "con"]),
            "rounds": max_rounds
        }
    }

    # Save result
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save complete result
        with open(output_dir / f"{claim_id}_step1_evidence.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"✓ 证据池已saved to: {output_dir / f'{claim_id}_step1_evidence.json'}")

    return result


async def process_claims_parallel(
    claims: List[Dict],
    max_rounds: int = 5,
    output_dir: Path = None,
    max_parallel: int = 8,
    start_index: int = 0
) -> List[Dict]:
    """
    并行Processing多个 claims

    Args:
        claims: claim 列表，每个元素为 {"claim": str, "verdict": str (可选)}
        max_rounds: 搜索轮次
        output_dir: 输出目录
        max_parallel: 最大并行数
        start_index: 起始索引（用于 claim_id 编号）

    Returns:
        结果列表
    """
    semaphore = asyncio.Semaphore(max_parallel)

    async def process_one(idx: int, claim_data: Dict):
        async with semaphore:
            claim_id = f"claim_{idx + start_index:04d}"

            # 断点续传：检查输出文件是否already exists
            if output_dir:
                output_file = output_dir / f"{claim_id}_step1_evidence.json"
                if output_file.exists():
                    print(f"\n⏭️  Skip {claim_id}（already exists）: {claim_data['claim'][:50]}...")
                    try:
                        with open(output_file, "r", encoding="utf-8") as f:
                            result = json.load(f)
                        # 确保包含 ground_truth
                        if "verdict" in claim_data and "ground_truth" not in result:
                            result["ground_truth"] = claim_data["verdict"]
                            # 重新保存以更新 ground_truth
                            with open(output_file, "w", encoding="utf-8") as f:
                                json.dump(result, f, ensure_ascii=False, indent=2)
                        return result
                    except Exception as e:
                        print(f"⚠ Reading已有文件failed: {e}，将重新Processing")

            print(f"\n{'#'*70}")
            print(f"StartProcessing {claim_id}: {claim_data['claim'][:50]}...")
            print(f"{'#'*70}")

            # 在线程池中运行同步函数
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                collect_evidence_for_claim,
                claim_data["claim"],
                max_rounds,
                output_dir,
                claim_id
            )

            # 添加 ground_truth
            if "verdict" in claim_data:
                result["ground_truth"] = claim_data["verdict"]
                # 重新保存以包含 ground_truth
                if output_dir:
                    output_file = output_dir / f"{claim_id}_step1_evidence.json"
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"✓ {claim_id} Processingcompleted")
            return result

    # 创建所有任务
    tasks = [process_one(i, claim_data) for i, claim_data in enumerate(claims)]

    # 并行执行
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Processing异常
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"❌ claim_{i + start_index:04d} Processingfailed: {result}")
            final_results.append({
                "claim_id": f"claim_{i + start_index:04d}",
                "claim": claims[i]["claim"],
                "error": str(result)
            })
        else:
            final_results.append(result)

    return final_results


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Step 1: Evidence Collection")
    parser.add_argument("--claim", type=str)
    parser.add_argument("--dataset", type=str, default="data/dataset_latest.json")
    parser.add_argument("--output", type=str, default="output_pipeline/step1_evidence", help="输出目录")
    parser.add_argument("--max-samples", type=int, default=None, help="最大Processing数量")
    parser.add_argument("--max-rounds", type=int, default=5, help="搜索轮次")
    parser.add_argument("--max-parallel", type=int, default=8, help="最大并行数")
    parser.add_argument("--start-index", type=int, default=0, help="起始索引")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.claim:
        # Processing单个 claim
        result = collect_evidence_for_claim(
            claim=args.claim,
            max_rounds=args.max_rounds,
            output_dir=output_dir,
            claim_id="single"
        )
        print(f"\n✓ 收集了 {result['statistics']['total_evidence']} 条证据")

    elif args.dataset:
        # Processing数据集
        with open(args.dataset, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if args.max_samples:
            dataset = dataset[:args.max_samples]

        print(f"\n{'='*80}")
        print(f"批量Processing模式")
        print(f"{'='*80}")
        print(f"数据集: {args.dataset}")
        print(f"样本数: {len(dataset)}")
        print(f"并行数: {args.max_parallel}")
        print(f"轮次: {args.max_rounds}")
        print(f"{'='*80}\n")

        # 并行Processing
        results = asyncio.run(process_claims_parallel(
            claims=dataset,
            max_rounds=args.max_rounds,
            output_dir=output_dir,
            max_parallel=args.max_parallel,
            start_index=args.start_index
        ))

        # 提取failed的数据
        failed_claims = []
        for i, result in enumerate(results):
            if "error" in result:
                failed_claims.append({
                    "claim_id": result.get("claim_id", f"claim_{i + args.start_index:04d}"),
                    "claim": result.get("claim", dataset[i]["claim"]),
                    "error": result["error"],
                    "original_index": i + args.start_index
                })
        
        # 保存汇总
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_claims": len(results),
            "successful": len([r for r in results if "error" not in r]),
            "failed": len(failed_claims),
            "results": results
        }

        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 保存failed列表（方便重新Processing）
        if failed_claims:
            failed_file = output_dir / "failed_claims.json"
            with open(failed_file, "w", encoding="utf-8") as f:
                json.dump(failed_claims, f, ensure_ascii=False, indent=2)
            print(f"\n⚠️  failed的数据已saved to: {failed_file}")
            print(f"   可以使用以下数据重新Processing:")
            for fc in failed_claims:
                print(f"   - {fc['claim_id']}: {fc['claim'][:50]}...")

        print(f"\n{'='*80}")
        print(f"Processingcompleted")
        print(f"{'='*80}")
        print(f"successful: {summary['successful']}/{summary['total_claims']}")
        print(f"failed: {summary['failed']}/{summary['total_claims']}")
        print(f"结果已saved to: {output_dir}")
        print(f"{'='*80}\n")

    else:
        # 默认测试
        test_claim = "小威廉姆斯是史上获得大满贯最多的女子网球运动员"
        result = collect_evidence_for_claim(
            claim=test_claim,
            max_rounds=2,
            output_dir=output_dir,
            claim_id="test"
        )
        print(f"\n✓ 测试completed，收集了 {result['statistics']['total_evidence']} 条证据")


if __name__ == "__main__":
    main()
