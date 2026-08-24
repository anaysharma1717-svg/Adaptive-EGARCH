"""
Regenerates the project's headline chart from already-computed results
(no re-fitting): crisis-day forecast loss (QLIKE) by model, showing raw
EGARCH's significant crisis-day edge over the HAR baseline (Section 3),
and how every bias-correction variant in the M2b->M2e chain (Section 4)
erodes that edge back toward the baseline.

Source numbers: research/results/05_regime_weighted_combination/m9_metrics.csv
                research/results/04_bias_correction_chain/m2c_metrics.csv
                research/results/04_bias_correction_chain/m2e_metrics.csv
                research/results/03_crisis_regime_test/crisis_dm_tests_A_and_B.csv
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- crisis-subsample QLIKE by model (N=38), read directly off the CSVs above ---
models = ["M1\nbaseline", "M2\nraw EGARCH", "M2b\npooled fix", "M2c\nregime fix", "M2d\nQLIKE-fit fix", "M2e\nno fix in crisis"]
qlike =  [1.4913,          0.3686,          1.1315,           1.1136,           0.8397,               0.7945]

# categorical: gray = neutral baseline, orange = the raw-EGARCH discovery, blue = correction chain
COLOR_BASELINE  = "#c3c2b7"
COLOR_DISCOVERY = "#eb6834"
COLOR_CHAIN     = "#2a78d6"
INK_PRIMARY     = "#0b0b0b"
INK_SECONDARY   = "#52514e"
INK_MUTED       = "#898781"
GRID            = "#e1e0d9"
SURFACE         = "#fcfcfb"

colors = [COLOR_BASELINE, COLOR_DISCOVERY, COLOR_CHAIN, COLOR_CHAIN, COLOR_CHAIN, COLOR_CHAIN]

plt.rcParams["font.family"] = "Segoe UI" if "Segoe UI" in {f.name for f in fm.fontManager.ttflist} else "DejaVu Sans"

fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

x = range(len(models))
bars = ax.bar(x, qlike, color=colors, width=0.62, zorder=3)

# baseline reference line
ax.axhline(qlike[0], color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
ax.text(len(models) - 0.42, qlike[0] + 0.035, "M1 baseline level", color=INK_MUTED, fontsize=9.5, ha="right")

# direct value labels
for i, (rect, v) in enumerate(zip(bars, qlike)):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.035, f"{v:.2f}",
             ha="center", va="bottom", fontsize=11.5, color=INK_PRIMARY, fontweight="bold")

# annotate the significant crisis-day DM test on the M2 raw bar
ax.annotate("DM = +2.80, p = 0.008\nvs. M1 baseline (N=38)",
            xy=(1, qlike[1]), xytext=(1, qlike[1] - 0.62),
            ha="center", va="top", fontsize=9.5, color=COLOR_DISCOVERY, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=COLOR_DISCOVERY, linewidth=1))

# bracket over the correction chain
chain_y = max(qlike[2:]) + 0.16
ax.plot([1.85, 1.85, 5.15, 5.15], [chain_y - 0.05, chain_y, chain_y, chain_y - 0.05],
        color=COLOR_CHAIN, linewidth=1.2, zorder=2)
ax.text(3.5, chain_y + 0.05, "every bias-correction tried pulls the edge back toward baseline",
        ha="center", va="bottom", fontsize=10, color=COLOR_CHAIN, fontweight="bold")

ax.set_xticks(list(x))
ax.set_xticklabels(models, fontsize=10.5, color=INK_SECONDARY)
ax.set_ylabel("QLIKE loss on crisis-regime days  (lower = better)", fontsize=11, color=INK_SECONDARY)
ax.set_ylim(0, 1.85)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9.5)
ax.tick_params(axis="x", length=0)
ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

fig.suptitle("Raw EGARCH has a real crisis-day edge — and every standard fix for its bias erases it",
             fontsize=14.5, fontweight="bold", color=INK_PRIMARY, x=0.01, ha="left", y=0.985)
ax.set_title("Top-decile realized-volatility days, out-of-sample walk-forward test (N=38)  ·  SPY, 2025-2026",
             fontsize=10.5, color=INK_MUTED, loc="left", pad=14)

fig.text(0.01, 0.005,
         "Adaptive EGARCH — SPY volatility forecasting  ·  github.com/anaysharma1717-svg/Adaptive-EGARCH",
         fontsize=8.5, color=INK_MUTED, ha="left")

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
out_path = "research/results/headline_crisis_edge.png"
fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
print(f"saved {out_path}")
