"""
Positive control figure (new Figure 4).

Reads only the already-computed positive-control draws; recomputes no statistic.

(a) Recall — how often the planted pathway is recovered — against spike level,
    for the targets testable under both backgrounds, so the two arms are
    compared on the same pathways.
(b) Pathways reported other than the planted one. At k = 0 nothing is planted,
    so every one of these is false by construction; at higher k some reflect
    genuine overlap between Reactome pathways, so k = 0 is the clean read.
"""
import pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).resolve().parent
ENR = _HERE.parent / "analysis" / "results" / "enrichment_tables"
FIGDIR = _HERE / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
C_PANEL, C_GENOME = "#2c6fb5", "#c1442e"

m = pd.read_csv(ENR / "fix11_positive_control_matched_summary_reactome.csv")

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
XLAB = "Planted pathway genes in the 40-gene query (k)"

ax = axes[0]
ax.plot(m.spike_k, 100 * m.panel_recall, "o-", color=C_PANEL, label="Panel-specific")
ax.plot(m.spike_k, 100 * m.genome_recall, "s--", color=C_GENOME, label="Genome-wide")
ax.set_xlabel(XLAB)
ax.set_ylabel("Planted pathway recovered (%)")
ax.set_title("(a) Real enrichment is recovered", fontsize=9.5, loc="left")
ax.set_ylim(-4, 108)
ax.set_xticks(list(m.spike_k))
ax.legend(loc="upper left", fontsize=8)

ax = axes[1]
ax.plot(m.spike_k, m.panel_other, "o-", color=C_PANEL, label="Panel-specific")
ax.plot(m.spike_k, m.genome_other, "s--", color=C_GENOME, label="Genome-wide")
ax.set_xlabel(XLAB)
ax.set_ylabel("Other pathways called significant")
ax.set_title("(b) Alongside far less noise", fontsize=9.5, loc="left")
ax.set_xticks(list(m.spike_k))
ax.set_ylim(-2, 40)
ax.annotate("k = 0: nothing planted,\nevery hit is false",
            xy=(0, m.genome_other.iloc[0]), xytext=(2.6, 30),
            fontsize=7.5, color="#444444",
            arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))
ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0.0, 0.80))

fig.tight_layout()
out = FIGDIR / "Figure4_positive_control.png"
fig.savefig(out)
print("wrote", out.name)

k0 = m[m.spike_k == 0].iloc[0]
k15 = m[m.spike_k == 15].iloc[0]
print(f"  k=0  recall panel {k0.panel_recall:.3f} genome {k0.genome_recall:.3f} | "
      f"other panel {k0.panel_other:.3f} genome {k0.genome_other:.3f}")
print(f"  k=15 recall panel {k15.panel_recall:.3f} genome {k15.genome_recall:.3f} | "
      f"other panel {k15.panel_other:.2f} genome {k15.genome_other:.2f}")
