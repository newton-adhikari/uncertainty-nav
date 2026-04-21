# Paper figures for uncertainty calibration and failure prediction

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "experiments/results"
PLOTS_DIR = "experiments/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "figure.dpi": 150,
})

COLORS = {
    "ensemble": "#2196F3", "mc_dropout": "#E91E63", "vanilla": "#F44336",
    "lstm": "#FF9800", "gru": "#9C27B0", "large_mlp": "#4CAF50",
}
LABELS = {
    "ensemble": "Ensemble (N=5)", "mc_dropout": "MC-Dropout",
    "vanilla": "Vanilla MLP", "lstm": "LSTM", "gru": "GRU",
    "large_mlp": "Large MLP",
}


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _find_mc(env, T=20):
    # Find MC-Dropout result, trying T-specific filename first
    for p in [f"{RESULTS_DIR}/mc_dropout_T{T}_env{env}.json",
              f"{RESULTS_DIR}/mc_dropout_env{env}.json"]:
        d = _load(p)
        if d:
            return d
    return None


def _save(name):
    plt.savefig(f"{PLOTS_DIR}/{name}.pdf", bbox_inches="tight")
    plt.savefig(f"{PLOTS_DIR}/{name}.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {name}")
    plt.close()



# Fig 1: Method comparison across 4 environments
def fig1_method_comparison():
    methods = ["vanilla", "lstm", "gru", "large_mlp", "mc_dropout", "ensemble"]
    envs = ["A", "C", "D", "B"]
    env_labels = ["Env A\n(train)", "Env C\n(sensor shift)", "Env D\n(layout shift)", "Env B\n(combined)"]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(envs))
    n = len(methods)
    width = 0.13

    for i, m in enumerate(methods):
        vals, errs = [], []
        for env in envs:
            if m == "mc_dropout":
                d = _find_mc(env)
            else:
                d = _load(f"{RESULTS_DIR}/{m}_env{env}.json")
            vals.append(d["success_rate"] if d else 0)
            errs.append(d.get("success_rate_std", 0) if d else 0)
        offset = (i - n/2 + 0.5) * width
        ax.bar(x + offset, vals, width, yerr=errs, capsize=3,
               color=COLORS[m], alpha=0.85, label=LABELS[m])

    ax.set_xticks(x)
    ax.set_xticklabels(env_labels)
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.15)
    ax.set_title("Navigation Performance Across Distribution Shift Spectrum")
    ax.legend(ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save("fig1_method_comparison")



# Fig 2: Calibration curve — SR per quartile (both methods, Env B)

def fig2_calibration():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    q_labels = ["Q0\n(low unc)", "Q1", "Q2", "Q3\n(high unc)"]
    q_colors = ["#4CAF50", "#8BC34A", "#FF9800", "#F44336"]

    for ax, (label, loader) in zip(axes, [
        ("Ensemble (Env B)", lambda: _load(f"{RESULTS_DIR}/ensemble_envB.json")),
        ("MC-Dropout (Env B)", lambda: _find_mc("B")),
    ]):
        d = loader()
        if not d or "uncertainty_calibration" not in d:
            ax.set_title(f"{label} — no data")
            continue
        cal = d["uncertainty_calibration"]
        srs = [cal.get(f"q{b}_sr", 0) for b in range(4)]
        uncs = [cal.get(f"q{b}_mean_unc", 0) for b in range(4)]

        bars = ax.bar(range(4), srs, color=q_colors, alpha=0.85)
        ax.set_xticks(range(4))
        ax.set_xticklabels(q_labels)
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1.15)
        ax.set_title(f"Uncertainty Calibration — {label}")
        ax.grid(axis="y", alpha=0.3)
        for i, (bar, sr, unc) in enumerate(zip(bars, srs, uncs)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{sr:.2f}\n(u={unc:.3f})", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    _save("fig2_calibration")



# Fig 3: AUROC comparison across environments

def fig3_failure_prediction():
    envs = ["A", "C", "D", "B"]
    env_labels = ["Env A", "Env C", "Env D", "Env B"]

    ens_auroc, mc_auroc = [], []
    for env in envs:
        d_ens = _load(f"{RESULTS_DIR}/ensemble_env{env}.json")
        d_mc = _find_mc(env)
        ens_auroc.append(d_ens.get("auroc_failure", 0) if d_ens else 0)
        mc_auroc.append(d_mc.get("auroc_failure", 0) if d_mc else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(envs))
    w = 0.35
    ax.bar(x - w/2, ens_auroc, w, color=COLORS["ensemble"], alpha=0.85, label="Ensemble")
    ax.bar(x + w/2, mc_auroc, w, color=COLORS["mc_dropout"], alpha=0.85, label="MC-Dropout")

    for i in range(len(envs)):
        ax.text(x[i] - w/2, ens_auroc[i] + 0.01, f"{ens_auroc[i]:.3f}", ha="center", fontsize=9)
        ax.text(x[i] + w/2, mc_auroc[i] + 0.01, f"{mc_auroc[i]:.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(env_labels)
    ax.set_ylabel("AUROC (Failure Prediction)")
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random (0.5)")
    ax.set_title("Failure Prediction Quality: Ensemble vs MC-Dropout")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save("fig3_failure_prediction")



# Fig 4: Routing table — autonomous SR vs human burden

def fig4_routing():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, env, env_label in zip(axes, ["B", "D"], ["Env B (combined)", "Env D (layout)"]):
        d_ens = _load(f"{RESULTS_DIR}/ensemble_env{env}.json")
        d_mc = _find_mc(env)

        for d, label, color, marker in [
            (d_ens, "Ensemble", COLORS["ensemble"], "o"),
            (d_mc, "MC-Dropout", COLORS["mc_dropout"], "s"),
        ]:
            if not d or "routing_table" not in d:
                continue
            rt = d["routing_table"]
            burdens = [r["human_burden"] for r in rt]
            auto_srs = [r["autonomous_sr"] for r in rt]
            ax.plot(burdens, auto_srs, f"-{marker}", color=color, linewidth=2,
                    markersize=8, label=label)
            for b, sr in zip(burdens, auto_srs):
                ax.annotate(f"{sr:.2f}", (b, sr), textcoords="offset points",
                            xytext=(5, 5), fontsize=8)

        ax.set_xlabel("Human Burden (fraction routed to human)")
        ax.set_ylabel("Autonomous SR (on retained subset)")
        ax.set_title(f"Selective Deployment — {env_label}")
        ax.set_xlim(-0.05, 0.85)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    _save("fig4_routing")



# Fig 5: Ensemble size ablation — AUROC vs N

def fig5_ensemble_size():
    data = _load(f"{RESULTS_DIR}/ensemble_size_auroc.json")
    if not data:
        print("  Skipping fig5 — no ensemble_size_auroc.json")
        return

    ns, srs, aurocs = [], [], []
    for key in sorted(data.keys(), key=lambda k: int(k.split("=")[1])):
        v = data[key]
        ns.append(int(key.split("=")[1]))
        srs.append(v["success_rate"])
        aurocs.append(v.get("auroc", 0.5))

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    ax1.bar(range(len(ns)), srs, color=COLORS["ensemble"], alpha=0.5, label="SR")
    ax2.plot(range(len(ns)), aurocs, "r-o", linewidth=2, markersize=8, label="AUROC")
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    ax1.set_xticks(range(len(ns)))
    ax1.set_xticklabels([f"N={n}" for n in ns])
    ax1.set_ylabel("Success Rate", color=COLORS["ensemble"])
    ax2.set_ylabel("AUROC", color="red")
    ax1.set_ylim(0, 0.5)
    ax2.set_ylim(0.4, 1.0)
    ax1.set_title("Ensemble Size: N=5 Maximizes AUROC")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save("fig5_ensemble_size")



# Fig 6: OOD detection — uncertainty distribution across envs

def fig6_ood_detection():
    envs = ["A", "C", "D", "B"]
    env_labels = ["Env A (train)", "Env C (sensor)", "Env D (layout)", "Env B (combined)"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for env, label, color in zip(envs, env_labels, colors):
        d = _load(f"{RESULTS_DIR}/ensemble_env{env}.json")
        if not d:
            continue
        unc = d["mean_uncertainty"]
        ax.barh(label, unc, color=color, alpha=0.8)
        ax.text(unc + 0.01, label, f"{unc:.3f}", va="center", fontsize=10)

    ax.set_xlabel("Mean Epistemic Uncertainty")
    ax.set_title("Distribution Shift Detection: Uncertainty Tracks Difficulty")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save("fig6_ood_detection")



# Fig 7: MC-Dropout T ablation (if data exists)

def fig7_mc_t_ablation():
    ts, aurocs, srs = [], [], []
    for T in [5, 10, 20]:
        d = _load(f"{RESULTS_DIR}/mc_dropout_T{T}_envB.json")
        if d:
            ts.append(T)
            aurocs.append(d.get("auroc_failure", 0))
            srs.append(d["success_rate"])

    if len(ts) < 2:
        print("  Skipping fig7 — need at least 2 T values")
        return

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    ax1.bar(range(len(ts)), srs, color=COLORS["mc_dropout"], alpha=0.5, label="SR")
    ax2.plot(range(len(ts)), aurocs, "b-o", linewidth=2, markersize=8, label="AUROC")

    ax1.set_xticks(range(len(ts)))
    ax1.set_xticklabels([f"T={t}" for t in ts])
    ax1.set_ylabel("Success Rate", color=COLORS["mc_dropout"])
    ax2.set_ylabel("AUROC", color="blue")
    ax1.set_ylim(0, 0.6)
    ax2.set_ylim(0.8, 1.0)
    ax1.set_title("MC-Dropout: Inference Samples T vs Quality (Env B)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save("fig7_mc_t_ablation")



# Fig 8: Robustness curve (SR vs noise, Env A layout)

def fig8_robustness():
    methods = ["ensemble", "vanilla", "large_mlp"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for m in methods:
        d = _load(f"{RESULTS_DIR}/{m}_envA.json")
        if not d or "robustness_curve" not in d:
            continue
        rc = d["robustness_curve"]
        sigmas = sorted(rc.keys(), key=float)
        srs = [rc[s] for s in sigmas]
        ax.plot([float(s) for s in sigmas], srs, "-o", color=COLORS[m],
                linewidth=2, markersize=6, label=LABELS[m])

    ax.set_xlabel("Laser Noise σ")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Robustness to Sensor Noise (Env A Layout)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save("fig8_robustness")



if __name__ == "__main__":
    print("Generating figures...")
    fig1_method_comparison()
    fig2_calibration()
    fig3_failure_prediction()
    fig4_routing()
    fig5_ensemble_size()
    fig6_ood_detection()
    fig7_mc_t_ablation()
    fig8_robustness()
    print(f"\nAll plots saved to {PLOTS_DIR}/")
