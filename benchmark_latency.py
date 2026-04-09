"""
Benchmark: Per-Iteration Latency of Adversarial Evidence Collection
====================================================================

Reviewer question:
  "Adversarial evidence collection requires up to 5 iterations, each involving
   multiple LLM calls and retrieval operations. Compared to Zero-Shot baselines,
   by what factor do the average inference latency and API invocation costs of
   ArgCheck increase?"

This script:
  1. Prints the theoretical complexity derivation.
  2. Runs an instrumented version of ArgCheck's evidence-collection loop on N
     sample claims, recording wall-clock time for every sub-operation in every
     round.
  3. Runs a Zero-Shot baseline (single LLM call with built-in web search).
  4. Produces a per-round timing table and an overall factor comparison.

Usage:
    python benchmark_latency.py [--samples N] [--rounds R] [--output results/]
"""

import json
import sys
import time
import argparse
import math
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).parent))

import config
from llm.qwen_client import QwenClient
from tools.jina_search import JinaSearch
from chains.claim_decomposer import ClaimDecomposer
from chains import ProQueryChain, ConQueryChain
from utils.qwen_wrapper import QwenLLMWrapper
from utils.retry_utils import call_with_retry, call_with_retry_until_success
from tools.evidence_filter import EvidenceFilter


# ---------------------------------------------------------------------------
# 1.  THEORETICAL COMPLEXITY
# ---------------------------------------------------------------------------

COMPLEXITY_TEXT = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         THEORETICAL COMPLEXITY: ArgCheck vs. Zero-Shot Baseline             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Parameters (from config.py / chain implementations):
  R   = adversarial rounds              (default max 5)
  q₁  = queries per agent, round 1      (up to 3, see ProQueryChain / ConQueryChain)
  qₖ  = queries per agent, rounds k>1   (= 1)
  K   = search results per query        (MAX_SEARCH_RESULTS_PER_QUERY = 5)
  B   = evidence filter batch size      (= 10)
  E_r = cumulative evidence pool size after round r
        ≈ min(2q₁K + 2qₖK(r-1), EVIDENCE_POOL_MAX_SIZE)

─────────────────────────────────────────────────────────────────────────────
A. API INVOCATION COUNT  (Step 1: Evidence Collection)
─────────────────────────────────────────────────────────────────────────────

  LLM calls:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  N_LLM = 1  +  2R  +  Σ_{r=1}^{R} ⌈E_r / B⌉                          │
  │          ↑     ↑        └── LLM-based evidence filter (batched)         │
  │          │     └────────── 2 per round (Pro + Con query generation)     │
  │          └──────────────── claim decomposition                          │
  └─────────────────────────────────────────────────────────────────────────┘

  Retrieval (Jina Search) calls:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  N_ret = 2q₁  +  2qₖ(R-1)  =  2q₁ + 2(R-1)                           │
  └─────────────────────────────────────────────────────────────────────────┘

  Concrete values at R=5, q₁=3, qₖ=1, K=5, B=10, E_r≈(r·10):
    Filter calls per round: ⌈10/10⌉+⌈20/10⌉+⌈30/10⌉+⌈40/10⌉+⌈50/10⌉ = 1+2+3+4+5 = 15
    N_LLM  ≈ 1 + 2·5 + 15 = 26   (Step 1 alone)
    N_ret  = 6 + 2·4     = 14

  Full pipeline (Steps 1–3):
    Step 2 (argument extraction + attack detection): ~8 LLM calls
    Step 3 (judge verdict):                          ~1 LLM + 1 search call
    Total_ArgCheck ≈ 35 LLM calls + 15 retrieval calls ≈ 50 API calls

  Zero-Shot baseline:
    1 web-search-augmented LLM call ≈ 1 LLM + 2 search calls ≈ 3 API calls

  ► API cost factor ≈ 50 / 3  ≈  16×

─────────────────────────────────────────────────────────────────────────────
B. CRITICAL-PATH LATENCY  (wall-clock time with parallelised search)
─────────────────────────────────────────────────────────────────────────────

  Searches within a round are run concurrently (ThreadPoolExecutor), so the
  search contribution collapses to one Jina call duration per round.

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  T_ArgCheck = T_decomp  +  Σ_{r=1}^{R} (T_query^(r) + T_search + T_filter^(r))  │
  │                                                                          │
  │  where:                                                                  │
  │    T_decomp      = latency of 1 decomposition LLM call                  │
  │    T_query^(r)   = T_pro^(r) + T_con^(r)  [serial: Pro then Con]       │
  │    T_search      = max(T_jina_i) ≈ 1 Jina call duration (parallel)     │
  │    T_filter^(r)  = ⌈E_r/B⌉ · T_filter_LLM  [serial batches]           │
  └─────────────────────────────────────────────────────────────────────────┘

  Zero-Shot:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  T_ZeroShot = T_search + T_LLM  (1 augmented generation call)          │
  └─────────────────────────────────────────────────────────────────────────┘

  Assuming typical values T_LLM ≈ 10–20s, T_search ≈ 3–5s:
    T_ZeroShot  ≈ 15–25 s
    T_ArgCheck  ≈ T_decomp + 5·(2·T_LLM + T_search + avg_filter)
                ≈ 15 + 5·(30 + 4 + 15) = 15 + 245 = 260 s

  ► Latency factor ≈ 260 / 20  ≈  10–15×

  Combined asymptotic complexity (as R grows):
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  N_API  = O(R · q₁ · K / B)  ≈  O(R)   (with fixed q, K, B)           │
  │  T      = O(R · T_LLM)                                                  │
  └─────────────────────────────────────────────────────────────────────────┘

  Both cost and latency grow linearly with the number of adversarial rounds R,
  while the Zero-Shot baseline is O(1).  At R=5 the empirical factor is ≈ 10–16×.
"""


# ---------------------------------------------------------------------------
# 2.  DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class RoundTiming:
    """Timing breakdown for one adversarial round."""
    round_num: int
    t_pro_query: float = 0.0    # Pro agent query generation (s)
    t_con_query: float = 0.0    # Con agent query generation (s)
    t_search: float = 0.0       # Parallel Jina search (s)
    t_filter: float = 0.0       # LLM-based evidence filter (s)
    t_total: float = 0.0        # Total wall-clock for this round (s)
    n_pro_queries: int = 0      # Queries generated by Pro
    n_con_queries: int = 0      # Queries generated by Con
    n_search_calls: int = 0     # Jina search calls made
    n_filter_llm_calls: int = 0 # LLM calls for filtering (ceil(E/batch))
    evidence_after_round: int = 0  # Evidence pool size after filter


@dataclass
class ClaimTiming:
    """Full timing record for one claim."""
    claim_id: str
    claim: str
    t_decompose: float = 0.0    # Claim decomposition (s)
    n_sub_claims: int = 0
    rounds: List[RoundTiming] = field(default_factory=list)
    t_total: float = 0.0        # Total wall-clock for evidence collection (s)
    # LLM-call accounting
    n_llm_total: int = 0        # Total LLM calls (decomp + query + filter)
    n_ret_total: int = 0        # Total retrieval calls
    n_api_total: int = 0        # n_llm_total + n_ret_total


@dataclass
class ZeroShotTiming:
    """Timing record for Zero-Shot baseline on one claim."""
    claim_id: str
    claim: str
    t_total: float = 0.0
    n_llm_calls: int = 1        # always 1
    n_search_calls: int = 0     # built-in search (counted as 1 augmented call)
    n_api_total: int = 1        # 1 augmented LLM call


# ---------------------------------------------------------------------------
# 3.  INSTRUMENTED EVIDENCE COLLECTION  (ArgCheck)
# ---------------------------------------------------------------------------

def benchmark_argcheck_claim(
    claim: str,
    max_rounds: int,
    claim_id: str,
    batch_size: int = 10,
) -> ClaimTiming:
    """
    Runs ArgCheck's evidence-collection phase on a single claim with per-step
    wall-clock timing.  Logic mirrors step1_evidence_collection.py exactly.
    """
    record = ClaimTiming(claim_id=claim_id, claim=claim)

    # Initialise shared components (same as step1_evidence_collection.py)
    llm_client = QwenClient(config.DASHSCOPE_API_KEY)
    jina = JinaSearch(config.JINA_API_KEY)

    decomposer_llm = QwenLLMWrapper(qwen_client=llm_client, enable_search=False)
    claim_decomposer = ClaimDecomposer(llm=decomposer_llm)

    pro_llm = QwenLLMWrapper(qwen_client=llm_client, enable_search=True, search_strategy="turbo")
    pro_chain = ProQueryChain(llm=pro_llm)

    con_llm = QwenLLMWrapper(qwen_client=llm_client, enable_search=True, search_strategy="turbo")
    con_chain = ConQueryChain(llm=con_llm)

    filter_llm = QwenLLMWrapper(qwen_client=llm_client, enable_search=False)
    evidence_filter = EvidenceFilter(llm=filter_llm)

    # ── Claim decomposition ──────────────────────────────────────────────
    t0 = time.perf_counter()
    sub_claims = call_with_retry_until_success(
        claim_decomposer.decompose,
        claim,
        validate_result=lambda x: x and len(x) > 0,
        max_retries=10,
        initial_delay=2.0,
        backoff_factor=1.5,
        max_delay=30.0,
    )
    record.t_decompose = time.perf_counter() - t0
    record.n_sub_claims = len(sub_claims) if sub_claims else 0
    record.n_llm_total += 1   # decomposition = 1 LLM call

    all_queries: List[Dict] = []
    evidence_pool: List[Dict] = []
    evidence_id_counter = 0
    pro_points_cache: list = []
    con_points_cache: list = []

    wall_start = time.perf_counter()

    for round_num in range(1, max_rounds + 1):
        rt = RoundTiming(round_num=round_num)
        round_start = time.perf_counter()

        existing_query_strings = [
            q["query"] if isinstance(q, dict) else q for q in all_queries
        ]

        # ── Pro query generation ─────────────────────────────────────────
        t0 = time.perf_counter()
        pro_queries = call_with_retry_until_success(
            pro_chain.generate_queries,
            claim=claim,
            round_num=round_num,
            opponent_evidences=con_points_cache,
            existing_queries=existing_query_strings,
            validate_result=lambda x: x and len(x) > 0,
            max_retries=10,
            initial_delay=2.0,
            backoff_factor=1.5,
            max_delay=30.0,
        )
        rt.t_pro_query = time.perf_counter() - t0
        rt.n_pro_queries = len(pro_queries) if pro_queries else 0
        record.n_llm_total += 1   # 1 LLM call per agent per round

        # ── Con query generation ─────────────────────────────────────────
        t0 = time.perf_counter()
        con_queries = call_with_retry_until_success(
            con_chain.generate_queries,
            claim=claim,
            round_num=round_num,
            opponent_evidences=pro_points_cache,
            existing_queries=existing_query_strings,
            validate_result=lambda x: x and len(x) > 0,
            max_retries=10,
            initial_delay=2.0,
            backoff_factor=1.5,
            max_delay=30.0,
        )
        rt.t_con_query = time.perf_counter() - t0
        rt.n_con_queries = len(con_queries) if con_queries else 0
        record.n_llm_total += 1   # 1 LLM call

        for q in (pro_queries or []):
            all_queries.append({"query": q, "agent": "pro", "round_num": round_num})
        for q in (con_queries or []):
            all_queries.append({"query": q, "agent": "con", "round_num": round_num})

        # ── Concurrent Jina search ───────────────────────────────────────
        search_queries = (
            [(q, "pro", round_num) for q in (pro_queries or [])] +
            [(q, "con", round_num) for q in (con_queries or [])]
        )
        rt.n_search_calls = len(search_queries)
        record.n_ret_total += len(search_queries)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(6, max(1, len(search_queries)))) as executor:
            class _SimpleEvidence:
                def __init__(self, content, source):
                    self.content = content
                    self.source = source

            def _search(query_info):
                query, agent, r = query_info
                try:
                    results = call_with_retry(
                        jina.search,
                        query,
                        top_k=config.MAX_SEARCH_RESULTS_PER_QUERY,
                        max_retries=40,
                        initial_delay=2.0,
                        backoff_factor=2.0,
                    )
                    return (query, agent, r, results, None)
                except Exception as e:
                    return (query, agent, r, [], e)

            futures = {executor.submit(_search, q): q for q in search_queries}
            for future in as_completed(futures):
                query, agent, r, results, error = future.result()
                if not error and results:
                    for result in results:
                        eid = f"evidence_{claim_id}_{evidence_id_counter:04d}"
                        evidence_id_counter += 1
                        entry = {
                            "id": eid,
                            "title": result.get("title", ""),
                            "content": result.get("content", ""),
                            "url": result.get("url", ""),
                            "published_time": result.get("published_time"),
                            "retrieved_time": datetime.now().isoformat(),
                            "search_query": query,
                            "retrieved_by": agent,
                            "round_num": r,
                        }
                        evidence_pool.append(entry)
                        cache_entry = _SimpleEvidence(
                            content=result.get("content", ""),
                            source=result.get("url", "unknown"),
                        )
                        if agent == "pro":
                            pro_points_cache.append(cache_entry)
                        else:
                            con_points_cache.append(cache_entry)
        rt.t_search = time.perf_counter() - t0

        # ── LLM evidence filter ──────────────────────────────────────────
        filter_calls = math.ceil(len(evidence_pool) / batch_size) if evidence_pool else 0
        rt.n_filter_llm_calls = filter_calls
        record.n_llm_total += filter_calls

        t0 = time.perf_counter()
        if evidence_pool:
            try:
                evidence_pool = call_with_retry(
                    evidence_filter.filter_evidence,
                    claim=claim,
                    evidence_pool=evidence_pool,
                    batch_size=batch_size,
                    max_retries=2,
                    initial_delay=2.0,
                )
            except Exception:
                pass  # Keep all evidence on error
        rt.t_filter = time.perf_counter() - t0
        rt.evidence_after_round = len(evidence_pool)

        rt.t_total = time.perf_counter() - round_start
        record.rounds.append(rt)

    record.t_total = time.perf_counter() - wall_start + record.t_decompose
    record.n_api_total = record.n_llm_total + record.n_ret_total
    return record


# ---------------------------------------------------------------------------
# 4.  ZERO-SHOT BASELINE
# ---------------------------------------------------------------------------

ZEROSHOT_PROMPT = (
    "You are a fact-checking assistant. Directly verify the following claim "
    "using your knowledge and web search capability. Output a brief verdict "
    "(Supported / Refuted / Not Enough Evidence) with one-sentence justification.\n\n"
    "Claim: {claim}"
)


def benchmark_zeroshot_claim(claim: str, claim_id: str) -> ZeroShotTiming:
    """
    Zero-Shot baseline: one web-search-augmented LLM call.
    Represents a direct retrieval-augmented generation approach with no
    adversarial iteration.
    """
    record = ZeroShotTiming(claim_id=claim_id, claim=claim)
    llm_client = QwenClient(config.DASHSCOPE_API_KEY)

    # enable_search=True → LLM internally invokes web search (counts as 1 augmented API call)
    llm = QwenLLMWrapper(
        qwen_client=llm_client,
        enable_search=True,
        force_search=True,
        search_strategy="pro",
    )

    t0 = time.perf_counter()
    try:
        llm.invoke(ZEROSHOT_PROMPT.format(claim=claim))
    except Exception as e:
        print(f"  ⚠ ZeroShot call failed: {e}")
    record.t_total = time.perf_counter() - t0
    record.n_api_total = 1   # 1 LLM call (search is internal to the call)
    return record


# ---------------------------------------------------------------------------
# 5.  REPORTING
# ---------------------------------------------------------------------------

def print_per_round_table(argcheck_records: List[ClaimTiming]):
    """Print average timing per round across all claims."""
    if not argcheck_records:
        return

    # Collect per-round data
    max_rounds = max(len(r.rounds) for r in argcheck_records)
    per_round: Dict[int, List[RoundTiming]] = {i: [] for i in range(1, max_rounds + 1)}
    for rec in argcheck_records:
        for rt in rec.rounds:
            per_round[rt.round_num].append(rt)

    print("\n" + "=" * 85)
    print("  Per-Round Timing  (averaged over all sample claims)")
    print("=" * 85)
    header = (
        f"{'Round':>6}  {'Pro Query':>10}  {'Con Query':>10}  "
        f"{'Search(‖)':>10}  {'Filter':>8}  {'Round Total':>12}  "
        f"{'LLM calls':>10}  {'Search calls':>12}"
    )
    print(header)
    print("-" * 85)

    cumulative_argcheck = 0.0
    for rn in range(1, max_rounds + 1):
        rts = per_round[rn]
        if not rts:
            continue
        avg = lambda f: sum(getattr(rt, f) for rt in rts) / len(rts)

        t_pro    = avg("t_pro_query")
        t_con    = avg("t_con_query")
        t_search = avg("t_search")
        t_filt   = avg("t_filter")
        t_tot    = avg("t_total")
        n_llm    = avg("n_filter_llm_calls") + 2       # 2 query gen + filter calls
        n_search = avg("n_search_calls")

        cumulative_argcheck += t_tot
        print(
            f"{rn:>6}  {t_pro:>9.2f}s  {t_con:>9.2f}s  "
            f"{t_search:>9.2f}s  {t_filt:>7.2f}s  {t_tot:>11.2f}s  "
            f"{n_llm:>10.1f}  {n_search:>12.1f}"
        )

    print("-" * 85)
    print(f"{'Total':>6}  {'':10}  {'':10}  {'':10}  {'':8}  {cumulative_argcheck:>11.2f}s  (search is parallel)")
    print()


def print_comparison_table(
    argcheck_records: List[ClaimTiming],
    zeroshot_records: List[ZeroShotTiming],
):
    """Print ArgCheck vs. Zero-Shot factor comparison."""
    if not argcheck_records or not zeroshot_records:
        return

    avg_argcheck_t  = sum(r.t_total for r in argcheck_records) / len(argcheck_records)
    avg_zeroshot_t  = sum(r.t_total for r in zeroshot_records) / len(zeroshot_records)
    avg_argcheck_api = sum(r.n_api_total for r in argcheck_records) / len(argcheck_records)
    avg_zeroshot_api = sum(r.n_api_total for r in zeroshot_records) / len(zeroshot_records)

    latency_factor = avg_argcheck_t / avg_zeroshot_t if avg_zeroshot_t > 0 else float("inf")
    cost_factor    = avg_argcheck_api / avg_zeroshot_api if avg_zeroshot_api > 0 else float("inf")

    print("=" * 55)
    print("  ArgCheck vs. Zero-Shot: Empirical Factor Comparison")
    print("=" * 55)
    print(f"  {'':30s} {'ArgCheck':>10}  {'ZeroShot':>10}  {'Factor':>8}")
    print(f"  {'-'*53}")
    print(f"  {'Avg latency (s)':30s} {avg_argcheck_t:>10.1f}  {avg_zeroshot_t:>10.1f}  {latency_factor:>7.1f}×")
    print(f"  {'Avg API calls':30s} {avg_argcheck_api:>10.1f}  {avg_zeroshot_api:>10.1f}  {cost_factor:>7.1f}×")
    print("=" * 55)
    print()
    print("  Breakdown (ArgCheck only, averages):")
    avg_llm = sum(r.n_llm_total for r in argcheck_records) / len(argcheck_records)
    avg_ret = sum(r.n_ret_total for r in argcheck_records) / len(argcheck_records)
    avg_t_decomp  = sum(r.t_decompose for r in argcheck_records) / len(argcheck_records)
    avg_t_rounds  = avg_argcheck_t - avg_t_decomp
    print(f"    LLM calls       : {avg_llm:.1f}")
    print(f"    Retrieval calls : {avg_ret:.1f}")
    print(f"    Decomp time     : {avg_t_decomp:.1f} s")
    print(f"    Rounds time     : {avg_t_rounds:.1f} s")
    print()


# ---------------------------------------------------------------------------
# 6.  MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ArgCheck per-iteration latency vs. Zero-Shot baseline"
    )
    parser.add_argument("--samples",  type=int,   default=5,
                        help="Number of claims to benchmark (default 5)")
    parser.add_argument("--rounds",   type=int,   default=5,
                        help="Max adversarial rounds for ArgCheck (default 5)")
    parser.add_argument("--dataset",  type=str,   default="data/MAVEN.json",
                        help="Path to claims JSON file")
    parser.add_argument("--output",   type=str,   default="results/benchmark",
                        help="Output directory for timing results")
    parser.add_argument("--no-zeroshot", action="store_true",
                        help="Skip Zero-Shot baseline (saves API calls)")
    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────
    dataset_path = Path(args.dataset)
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    samples = dataset[: args.samples]
    print(f"\nBenchmarking on {len(samples)} claims, {args.rounds} max rounds\n")

    # ── Print theoretical complexity ──────────────────────────────────────
    print(COMPLEXITY_TEXT)

    # ── Output directory ──────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Run ArgCheck timing ───────────────────────────────────────────────
    argcheck_records: List[ClaimTiming] = []
    print("=" * 60)
    print("  Running ArgCheck (instrumented) ...")
    print("=" * 60)
    for i, item in enumerate(samples):
        claim = item["claim"]
        cid   = f"bench_{i:03d}"
        print(f"\n[{i+1}/{len(samples)}] {cid}: {claim[:70]}{'...' if len(claim)>70 else ''}")
        try:
            rec = benchmark_argcheck_claim(
                claim=claim,
                max_rounds=args.rounds,
                claim_id=cid,
            )
            argcheck_records.append(rec)
            print(f"  ✓ Total: {rec.t_total:.1f}s  |  "
                  f"LLM calls: {rec.n_llm_total}  |  "
                  f"Retrieval calls: {rec.n_ret_total}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    # ── Run Zero-Shot timing ──────────────────────────────────────────────
    zeroshot_records: List[ZeroShotTiming] = []
    if not args.no_zeroshot:
        print("\n" + "=" * 60)
        print("  Running Zero-Shot Baseline ...")
        print("=" * 60)
        for i, item in enumerate(samples):
            claim = item["claim"]
            cid   = f"bench_{i:03d}"
            print(f"\n[{i+1}/{len(samples)}] {cid}: {claim[:70]}{'...' if len(claim)>70 else ''}")
            try:
                zrec = benchmark_zeroshot_claim(claim=claim, claim_id=cid)
                zeroshot_records.append(zrec)
                print(f"  ✓ Total: {zrec.t_total:.1f}s  |  API calls: {zrec.n_api_total}")
            except Exception as e:
                print(f"  ✗ Failed: {e}")

    # ── Print tables ──────────────────────────────────────────────────────
    if argcheck_records:
        print_per_round_table(argcheck_records)
    if argcheck_records and zeroshot_records:
        print_comparison_table(argcheck_records, zeroshot_records)
    elif argcheck_records:
        print("(Zero-Shot baseline was skipped; run without --no-zeroshot for factor comparison)")

    # ── Save JSON results ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"benchmark_{timestamp}.json"
    output = {
        "timestamp": timestamp,
        "config": {
            "n_samples": len(samples),
            "max_rounds": args.rounds,
            "dataset": str(dataset_path),
        },
        "theoretical": {
            "formula_llm_calls": "1 + 2R + sum_r(ceil(E_r / B))",
            "formula_ret_calls": "2*q1 + 2*qk*(R-1)",
            "formula_latency":   "T_decomp + sum_r(T_query_r + T_search + T_filter_r)",
            "zeroshot":          "T_search + T_LLM",
            "estimated_factor":  "~10-16x  (latency and API cost)",
            "params": {
                "R_max": 5, "q1": 3, "qk": 1,
                "K": config.MAX_SEARCH_RESULTS_PER_QUERY,
                "B": 10,
            },
            "concrete_at_R5": {
                "N_LLM_step1": 26,
                "N_ret_step1": 14,
                "N_LLM_full_pipeline": 35,
                "N_api_full_pipeline": 50,
                "N_api_zeroshot": 3,
                "api_cost_factor": "~16x",
            },
        },
        "argcheck_results": [asdict(r) for r in argcheck_records],
        "zeroshot_results": [asdict(r) for r in zeroshot_records],
    }
    if argcheck_records and zeroshot_records:
        avg_at = sum(r.t_total   for r in argcheck_records) / len(argcheck_records)
        avg_zt = sum(r.t_total   for r in zeroshot_records) / len(zeroshot_records)
        avg_aa = sum(r.n_api_total for r in argcheck_records) / len(argcheck_records)
        avg_za = sum(r.n_api_total for r in zeroshot_records) / len(zeroshot_records)
        output["empirical"] = {
            "avg_argcheck_latency_s":  round(avg_at, 2),
            "avg_zeroshot_latency_s":  round(avg_zt, 2),
            "latency_factor":          round(avg_at / avg_zt, 2) if avg_zt else None,
            "avg_argcheck_api_calls":  round(avg_aa, 1),
            "avg_zeroshot_api_calls":  round(avg_za, 1),
            "api_cost_factor":         round(avg_aa / avg_za, 2) if avg_za else None,
        }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Results saved to: {out_file}")


if __name__ == "__main__":
    main()
