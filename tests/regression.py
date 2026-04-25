"""
Regression test suite. Run after every code change.
Catches recall drops, latency regressions, and broken outputs.
Targets ~30-60 second total runtime.
"""
import sys
import time
import statistics
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from search_ivf import (
    search_bruteforce, search_ivf,
    texts, all_doc_vectors, NUM_DOCS, K
)


# ---- Test queries with expected behavior ----
# Pick queries we know work well, where regressions would be obvious.
REGRESSION_QUERIES = [
    "history of the manhattan project",
    "vitamin D deficiency symptoms",
    "weather forecast",
    "what causes thunderstorms",
]

# Thresholds tuned from Day 9-10 baseline. Tighten over time.
THRESHOLDS = {
    "bf_latency_max_ms":      200,    # brute force shouldn't blow up
    "ivf_latency_max_ms":     150,    # ivf at nprobe=8 should stay under this
    "ivf_recall_min":         0.75,   # at least 75% recall@10 (we got 87.5%)
    "min_results_returned":    10,    # always return 10 docs
}


class TestResult:
    """Tiny custom test result holder. Avoid pytest dependency for now."""
    def __init__(self):
        self.passed = []
        self.failed = []

    def assert_(self, condition, name, detail=""):
        if condition:
            self.passed.append(name)
            print(f"  ✓ {name} {detail}")
        else:
            self.failed.append((name, detail))
            print(f"  ✗ {name} {detail}")

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*60}")
        print(f"Results: {len(self.passed)}/{total} passed")
        if self.failed:
            print("FAILED:")
            for name, detail in self.failed:
                print(f"  - {name}: {detail}")
        print(f"{'='*60}")
        return len(self.failed) == 0


def warm_up():
    """First query is always slow due to lazy init. Throw away."""
    search_bruteforce(REGRESSION_QUERIES[0], top_k=10)
    search_ivf(REGRESSION_QUERIES[0], nprobe=8, top_k=10)


def run_brute_force_tests(result):
    """Brute force should always return 10 results in reasonable time."""
    print("\nBrute force tests:")
    times = []
    for q in REGRESSION_QUERIES:
        t0 = time.perf_counter()
        results = search_bruteforce(q, top_k=10)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)

        result.assert_(
            len(results) == THRESHOLDS["min_results_returned"],
            f"bf returns {THRESHOLDS['min_results_returned']} results",
            f"({q!r:40s} → {len(results)} results)"
        )

    median_lat = statistics.median(times)
    result.assert_(
        median_lat < THRESHOLDS["bf_latency_max_ms"],
        f"bf median latency under {THRESHOLDS['bf_latency_max_ms']}ms",
        f"(actual: {median_lat:.0f}ms)"
    )


def run_ivf_tests(result):
    """IVF at nprobe=8 should hit recall and latency targets."""
    print("\nIVF tests (nprobe=8):")
    times = []
    recalls = []

    for q in REGRESSION_QUERIES:
        # Ground truth from brute force
        bf_results = search_bruteforce(q, top_k=10)
        bf_ids = set(d for d, _ in bf_results)

        # Time IVF
        t0 = time.perf_counter()
        ivf_results, _, _ = search_ivf(q, nprobe=8, top_k=10)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)

        ivf_ids = set(d for d, _ in ivf_results)
        recall = len(bf_ids & ivf_ids) / 10
        recalls.append(recall)

    median_lat = statistics.median(times)
    avg_recall = statistics.mean(recalls)

    result.assert_(
        median_lat < THRESHOLDS["ivf_latency_max_ms"],
        f"ivf median latency under {THRESHOLDS['ivf_latency_max_ms']}ms",
        f"(actual: {median_lat:.0f}ms)"
    )
    result.assert_(
        avg_recall >= THRESHOLDS["ivf_recall_min"],
        f"ivf recall@10 above {THRESHOLDS['ivf_recall_min']:.0%}",
        f"(actual: {avg_recall:.1%})"
    )


def run_correctness_tests(result):
    """Sanity check: known queries return expected docs."""
    print("\nCorrectness tests:")

    # Manhattan Project query should rank a Manhattan Project doc #1
    bf = search_bruteforce("history of the manhattan project", top_k=1)
    top_text = texts[bf[0][0]].lower()
    result.assert_(
        "manhattan" in top_text,
        "Manhattan query → top result mentions Manhattan",
        f"(top: {top_text[:60]}...)"
    )


def main():
    print("=" * 60)
    print("Regression test suite")
    print(f"Corpus: {NUM_DOCS} docs, {all_doc_vectors.shape[0]} tokens, K={K}")
    print("=" * 60)

    print("\nWarming up...")
    warm_up()

    result = TestResult()
    run_brute_force_tests(result)
    run_ivf_tests(result)
    run_correctness_tests(result)

    success = result.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

def save_run(passed, total, recall, latency):
    """Append this run's numbers to a JSONL log for trend tracking."""
    import json, datetime, subprocess
    
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "unknown"
    
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "commit": commit,
        "passed": passed,
        "total": total,
        "ivf_recall": recall,
        "ivf_latency_ms": latency,
    }
    Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
    with open("benchmarks/results/regression_history.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
