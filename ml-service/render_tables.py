import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "evaluation_results/results_tables.png"

config_cols = ["Configuration", "Accuracy", "F1 (Anomaly)", "Macro-F1", "ROC-AUC"]
config_rows = [
    ["Rules-only baseline", "0.474", "0.470", "0.474", "—"],
    ["TabPFN combined",     "0.914", "0.943", "0.882", "0.979"],
    ["Hybrid (deployed)",   "0.940", "0.961", "0.914", "0.988"],
]

layer_cols = ["Feature set", "# feat", "Accuracy", "Precision (Anom)",
              "Recall (Anom)", "F1 (Anom)", "ROC-AUC"]
layer_rows = [
    ["Behavioural-only",   "13", "0.897", "0.942", "0.920", "0.931", "0.964"],
    ["Attack-signal-only", "7",  "0.759", "0.759", "1.000", "0.863", "0.645"],
    ["Combined",           "20", "0.914", "0.943", "0.943", "0.943", "0.979"],
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.2))
fig.suptitle("Behavioural Anomaly Detector — Evaluation Results\n"
             "386 rows (Normal=93, Anomaly=293), 50-row train cap, 116-row test",
             fontsize=12, fontweight="bold")

def style_table(ax, cols, rows, title, highlight_row=None):
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=8)
    t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.6)
    for (r, c), cell in t.get_celldata().items() if hasattr(t, "get_celldata") else t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
        elif highlight_row is not None and r == highlight_row:
            cell.set_facecolor("#d5f5e3"); cell.set_text_props(fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f4f6f7")
        cell.set_edgecolor("#bdc3c7")
        if c == 0 and r != 0:
            cell.set_text_props(ha="left"); cell.PAD = 0.04

style_table(ax1, config_cols, config_rows,
            "Configuration comparison  (best = Hybrid)", highlight_row=3)
style_table(ax2, layer_cols, layer_rows,
            "Feature-layer comparison  (behaviour carries the signal)")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
