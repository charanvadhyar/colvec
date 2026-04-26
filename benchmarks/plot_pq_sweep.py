"""
Plot the PQ M-sweep results from the JSON output.
Highlights the sweet spot (M=32) where compression is essentially free.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt

with open("benchmarks/results/pq_m_sweep.json") as f:
    data = json.load(f)["results"]

ms = [r["M"] for r in data]
compressions = [r["compression"] for r in data]
ndcgs = [r["ndcg_at_10"] for r in data]
recalls = [r["recall_at_100"] for r in data]
latencies = [r["median_latency_ms"] for r in data]
storages = [r["storage_mb"] for r in data]

# Reference: uncompressed baseline (Day 11)
BASELINE_NDCG = 0.6122
BASELINE_RECALL = 0.8666
BASELINE_LAT = 108
BASELINE_STORAGE = 465.2

# The sweet spot
SWEET_SPOT_M = 32
sweet_idx = ms.index(SWEET_SPOT_M)

# Two-panel figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# ---- Left: Quality vs compression ----
ax1.plot(compressions, ndcgs, "o-", color="steelblue", linewidth=2, markersize=10, zorder=2)

# Highlight the sweet spot
ax1.scatter(
    [compressions[sweet_idx]], [ndcgs[sweet_idx]],
    s=300, color="gold", edgecolor="darkorange", linewidth=2,
    label=f"sweet spot (M={SWEET_SPOT_M})", zorder=3,
)

# Baseline reference line
ax1.axhline(
    BASELINE_NDCG, color="gray", linestyle="--", alpha=0.7,
    label=f"uncompressed baseline ({BASELINE_NDCG:.3f})", zorder=1,
)

# Annotate each M value
for r in data:
    offset = (10, -4) if r["M"] != SWEET_SPOT_M else (12, 8)
    ax1.annotate(
        f"M={r['M']}",
        (r["compression"], r["ndcg_at_10"]),
        textcoords="offset points",
        xytext=offset,
        fontsize=10,
        fontweight="bold" if r["M"] == SWEET_SPOT_M else "normal",
    )

ax1.set_xlabel("Compression ratio (×)")
ax1.set_ylabel("nDCG@10")
ax1.set_xscale("log")
ax1.set_title("Quality vs compression on SciFact\n(higher is better)")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="lower left")

# ---- Right: Latency vs M ----
ax2.plot(ms, latencies, "s-", color="orange", linewidth=2, markersize=10, zorder=2)

# Highlight the sweet spot
ax2.scatter(
    [ms[sweet_idx]], [latencies[sweet_idx]],
    s=300, color="gold", edgecolor="darkorange", linewidth=2,
    label=f"sweet spot (M={SWEET_SPOT_M})", zorder=3,
)

# Baseline reference line
ax2.axhline(
    BASELINE_LAT, color="gray", linestyle="--", alpha=0.7,
    label=f"uncompressed baseline ({BASELINE_LAT} ms)", zorder=1,
)

ax2.set_xlabel("M (number of subquantizers)")
ax2.set_ylabel("Median latency (ms)")
ax2.set_title("Latency vs M (naive Python kernel)\n(lower is better — Day 16 will fix this)")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper left")
ax2.set_yscale("log")

plt.suptitle(
    "Product quantization sweep: M=32 gives 16× compression for free",
    fontsize=14, fontweight="bold", y=1.02,
)
plt.tight_layout()

out_path = Path("benchmarks/results/pq_m_sweep.png")
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"Saved {out_path}")

# ---- Print summary table for blog use ----
print("\nFor the blog post:\n")
print(f"{'M':>4} | {'compress':>9} | {'storage':>9} | {'nDCG@10':>8} | {'vs base':>8} | {'lat ms':>7}")
print("-" * 65)
for r, m in zip(data, ms):
    delta = r["ndcg_at_10"] - BASELINE_NDCG
    marker = " ←" if m == SWEET_SPOT_M else ""
    print(f"{m:>4} | {r['compression']:>7.1f}× | {r['storage_mb']:>6.1f} MB | "
          f"{r['ndcg_at_10']:>8.4f} | {delta:>+8.4f} | {r['median_latency_ms']:>7.0f}{marker}")
print("-" * 65)
print(f"{'BF':>4} | {'1×':>9} | {BASELINE_STORAGE:>6.1f} MB | "
      f"{BASELINE_NDCG:>8.4f} | {'  0.0000':>8} | {BASELINE_LAT:>7}")