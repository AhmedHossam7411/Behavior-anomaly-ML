import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = "evaluation_results"

MODELS = ["Behavioural-only", "Attack-signal-only", "Combined (deployed)"]

METRICS = [
    ("Accuracy",                  [0.919, 0.969, 0.988]),
    ("Precision (PPV)",           [0.900, 1.000, 0.952]),
    ("Recall / Sensitivity (TPR)",[0.429, 0.762, 0.952]),
    ("Specificity (TNR)",         [0.993, 1.000, 0.993]),
    ("NPV",                       [0.920, 0.965, 0.993]),
    ("F1-score",                  [0.581, 0.865, 0.952]),
    ("Balanced accuracy",         [0.711, 0.881, 0.973]),
    ("G-mean",                    [0.652, 0.873, 0.972]),
    ("MCC",                       [0.588, 0.858, 0.945]),
    ("Cohen's κ",            [0.542, 0.848, 0.945]),
    ("FPR",                       [0.007, 0.000, 0.007]),
    ("FNR",                       [0.571, 0.238, 0.048]),
    ("ROC-AUC",                   [0.952, 0.962, 1.000]),
    ("PR-AUC (AP)",               [0.770, 0.897, 0.998]),
    ("EER",                       [0.1016, 0.0728, 0.0036]),
]

CM = {
    "Behavioural-only":   (9, 12, 1, 138),
    "Attack-signal-only": (16, 5, 0, 139),
    "Combined (deployed)":(20, 1, 1, 138),
}

def style_table(ax, cols, rows, title, highlight_col=None, first_col_left=True):
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=8)
    t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9.5)
    t.scale(1, 1.5)
    cells = t.get_celld()
    for (r, c), cell in cells.items():
        if r == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
        elif highlight_col is not None and c == highlight_col:
            cell.set_facecolor("#d5f5e3"); cell.set_text_props(fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f4f6f7")
        cell.set_edgecolor("#bdc3c7")
        if first_col_left and c == 0 and r != 0:
            cell.set_text_props(ha="left"); cell.PAD = 0.04

def fig_table():
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.5, 8.2), gridspec_kw={"height_ratios": [5, 1]})
    fig.suptitle("Full classification metrics — 70% train / 30% test split\n"
                 "533 rows (Normal=464, Anomaly=69, 12.9%)  ·  test set n=160 (21 anomalies)",
                 fontsize=12, fontweight="bold")

    cols = ["Metric"] + MODELS
    rows = []
    for name, vals in METRICS:
        if name == "EER":
            rows.append([name] + [f"{v*100:.2f}%" for v in vals])
        else:
            rows.append([name] + [f"{v:.3f}" for v in vals])
    style_table(ax1, cols, rows, "", highlight_col=3)

    cm_cols = ["Model", "TP", "FN", "FP", "TN"]
    cm_rows = [[m, *map(str, CM[m])] for m in MODELS]
    style_table(ax2, cm_cols, cm_rows, "Confusion matrices (test set)", highlight_col=None)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"{OUTDIR}/paper_metrics_table_70_30.png"
    plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    print("wrote", out)

def fig_graph():
    bar_metrics = ["Accuracy", "Precision (PPV)", "Recall / Sensitivity (TPR)",
                   "F1-score", "Balanced accuracy", "MCC", "ROC-AUC"]
    short = ["Accuracy", "Precision", "Recall", "F1", "Bal. Acc", "MCC", "ROC-AUC"]
    lookup = {n: v for n, v in METRICS}
    data = np.array([lookup[m] for m in bar_metrics])

    x = np.arange(len(bar_metrics)); w = 0.26
    colors = ["#95a5a6", "#e67e22", "#27ae60"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, model in enumerate(MODELS):
        bars = ax.bar(x + (i - 1) * w, data[:, i], w, label=model, color=colors[i])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=10)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Score")
    ax.set_title("Behavioural Anomaly Detector — metric comparison by feature layer\n"
                 "(70/30 split, test n=160)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = f"{OUTDIR}/paper_metrics_graph_70_30.png"
    plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    print("wrote", out)

def fig_literature():
    lit_rows = [
        ["[2] BehaveFormer (transformer)", "Keystroke+IMU", "Authentication", "EER 1.80% (Aalto) / 2.95% (HuMIdb)"],
        ["[5] keyRecs benchmark (LGBM)", "Keystroke", "Authentication", "F1 0.80 ; lowest EER/FRR"],
        ["[8] ML/DL mouse dynamics", "Mouse", "Continuous auth", "Acc 85.73% (1D-CNN bin.) / 92.48% (ANN multi)"],
        ["[10] SapiMouse deep features", "Mouse", "Auth + bot detection", "AUC ~0.977 (CNN-LSTM, SapiMouse)"],
        ["THIS WORK — Combined (deployed)", "Mouse+Keyboard+Attack", "Anomaly / intrusion det.",
         "EER 0.36% · Acc 0.988 · F1 0.952 · AUC 1.000"],
    ]
    lit_cols = ["Study", "Modality", "Task", "Reported headline result"]

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1])
    fig.suptitle("Literature comparison — reference papers vs. this work",
                 fontsize=13, fontweight="bold")

    ax_tbl = fig.add_subplot(gs[0, :])
    style_table(ax_tbl, lit_cols, lit_rows, "")
    for (r, c), cell in ax_tbl.tables[0].get_celld().items():
        if r == len(lit_rows):
            cell.set_facecolor("#d5f5e3"); cell.set_text_props(fontweight="bold")

    ax1 = fig.add_subplot(gs[1, 0])
    eer_lbl = ["Ours\n(Combined)", "BehaveFormer\n(Aalto)", "BehaveFormer\n(HuMIdb)"]
    eer_val = [0.36, 1.80, 2.95]
    b = ax1.bar(eer_lbl, eer_val, color=["#27ae60", "#7f8c8d", "#7f8c8d"])
    for bar in b:
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                 f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=8)
    ax1.set_title("EER %  (lower = better)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("EER (%)"); ax1.grid(axis="y", linestyle=":", alpha=0.5); ax1.set_axisbelow(True)

    ax2 = fig.add_subplot(gs[1, 1])
    acc_lbl = ["Ours\n(Combined)", "MDPI mouse\n(1D-CNN bin.)", "MDPI mouse\n(ANN multi)"]
    acc_val = [98.8, 85.73, 92.48]
    b = ax2.bar(acc_lbl, acc_val, color=["#27ae60", "#7f8c8d", "#7f8c8d"])
    for bar in b:
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                 f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylim(0, 105)
    ax2.set_title("Accuracy %  (higher = better)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)"); ax2.grid(axis="y", linestyle=":", alpha=0.5); ax2.set_axisbelow(True)

    ax3 = fig.add_subplot(gs[1, 2])
    f1_lbl = ["Ours\n(Combined)", "keyRecs\n(LGBM)"]
    f1_val = [95.2, 80.0]
    b = ax3.bar(f1_lbl, f1_val, color=["#27ae60", "#7f8c8d"])
    for bar in b:
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                 f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    ax3.set_ylim(0, 105)
    ax3.set_title("F1 %  (higher = better)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("F1 (%)"); ax3.grid(axis="y", linestyle=":", alpha=0.5); ax3.set_axisbelow(True)

    fig.text(0.5, 0.005,
             "Note: references are per-user authentication on large public datasets; this work is population-level "
             "anomaly detection on 533 in-house rows. EER is the shared axis — comparison is indicative, not a head-to-head benchmark.",
             ha="center", fontsize=8, style="italic", color="#555555")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    out = f"{OUTDIR}/literature_comparison_70_30.png"
    plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    print("wrote", out)

def fig_confusion():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    fig.suptitle("Confusion matrices — 70/30 split, test set n=160 (21 anomalies / 139 normal)",
                 fontsize=12, fontweight="bold")
    for ax, model in zip(axes, MODELS):
        tp, fn, fp, tn = CM[model]
        mat = np.array([[tn, fp], [fn, tp]])
        ax.imshow(mat, cmap="Greens", vmin=0, vmax=mat.max())
        ax.set_title(model, fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred\nNormal", "Pred\nAnomaly"], fontsize=9)
        ax.set_yticklabels(["Actual\nNormal", "Actual\nAnomaly"], fontsize=9)
        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                v = mat[i, j]
                tc = "white" if v > mat.max() * 0.55 else "#1b2631"
                ax.text(j, i, f"{labels[i][j]}\n{v}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=tc)
        ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = f"{OUTDIR}/confusion_matrices_70_30.png"
    plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    print("wrote", out)

if __name__ == "__main__":
    fig_table()
    fig_graph()
    fig_literature()
    fig_confusion()
