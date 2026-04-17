# Paper figures for uncertainty calibration and failure prediction

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

RESULTS_DIR = "experiments/results"
PLOTS_DIR = "experiments/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# added consistent style
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "figure.dpi": 150,
})

COLORS = {
    "ensemble": "#2196F3", "vanilla": "#F44336", "lstm": "#FF9800",
    "gru": "#9C27B0", "large_mlp": "#4CAF50",
}
LABELS = {
    "ensemble": "Deep Ensemble (Ours)", "vanilla": "Vanilla PPO",
    "lstm": "LSTM", "gru": "GRU", "large_mlp": "Large MLP",
}


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _run_ensemble_episodes(env_cfg, n_episodes=200):
    # Run ensemble on env, return per-episode (uncertainty, success) pairs
    import torch
    sys.path.insert(0, "src/uncertainty_nav")
    from uncertainty_nav.models import DeepEnsemble
    from uncertainty_nav.nav_env import PartialObsNavEnv

    device = torch.device("cpu")
    env = PartialObsNavEnv(env_cfg, seed=42)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    paths = [f"checkpoints/ensemble_m{i}_policy.pt" for i in range(5)]
    existing = [p for p in paths if os.path.exists(p)]
    if not existing:
        return None, None, None
    policy = DeepEnsemble.from_checkpoints(existing, obs_dim, act_dim, device=device)

    uncs, successes, step_uncs = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_uncs = []
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                out = policy(obs_t)
                ep_uncs.append(out["epistemic_uncertainty"].item())
                action = out["action"]
            obs, _, term, trunc, info = env.step(action.squeeze(0).numpy())
            done = term or trunc
        uncs.append(np.mean(ep_uncs))
        successes.append(float(info.get("success", False)))
        step_uncs.append(ep_uncs)
    return np.array(uncs), np.array(successes), step_uncs


# Figure 1: OOD Detection — Uncertainty Distribution Shift
def fig1_ood_detection():
    # Histogram: uncertainty distribution on Env A vs Env B.
    # Shows the ensemble detects distribution shift
    from uncertainty_nav.nav_env import ENV_A, ENV_B

    uncs_a, _, _ = _run_ensemble_episodes(ENV_A, 200)
    uncs_b, _, _ = _run_ensemble_episodes(ENV_B, 200)
    if uncs_a is None or uncs_b is None:
        print("Skipping fig1 — no ensemble checkpoints")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(uncs_a.max(), uncs_b.max()) * 1.1, 40)
    ax.hist(uncs_a, bins=bins, alpha=0.6, color="#2196F3", label=f"Env A (train) μ={uncs_a.mean():.3f}", density=True)
    ax.hist(uncs_b, bins=bins, alpha=0.6, color="#F44336", label=f"Env B (unseen) μ={uncs_b.mean():.3f}", density=True)
    ax.set_xlabel("Mean Epistemic Uncertainty per Episode")
    ax.set_ylabel("Density")
    ax.set_title("Distribution Shift Detection via Epistemic Uncertainty")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/fig1_ood_detection.pdf", bbox_inches="tight")
    plt.savefig(f"{PLOTS_DIR}/fig1_ood_detection.png", dpi=150, bbox_inches="tight")
    print("Saved fig1_ood_detection")


# Figure 2: Calibration Curve (THE key result)
def fig2_calibration():
    # Bar chart: SR per uncertainty quartile on Env A and Env B.
    # Shows uncertainty reliably predicts failure
    data_a = _load_json(f"{RESULTS_DIR}/ensemble_envA.json")
    data_b = _load_json(f"{RESULTS_DIR}/ensemble_envB.json")
    if not data_a or not data_b:
        print("Skipping fig2 — missing ensemble results")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    quartile_labels = ["Q1\n(low unc)", "Q2", "Q3", "Q4\n(high unc)"]

    for ax, data, env_name in zip(axes, [data_a, data_b], ["Env A (Training)", "Env B (Unseen)"]):
        cal = data.get("uncertainty_calibration", {})
        if not cal:
            continue
        srs = [cal.get(f"q{b}_sr", 0.0) for b in range(4)]
        uncs = [cal.get(f"q{b}_mean_unc", 0.0) for b in range(4)]

        x = np.arange(4)
        bars = ax.bar(x, srs, color=["#4CAF50", "#8BC34A", "#FF9800", "#F44336"], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(quartile_labels)
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1.15)
        ax.set_title(f"Uncertainty Calibration — {env_name}")
        ax.grid(axis="y", alpha=0.3)

        # Annotate bars with SR and uncertainty values
        for i, (bar, sr, unc) in enumerate(zip(bars, srs, uncs)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"SR={sr:.2f}\nσ={unc:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/fig2_calibration.pdf", bbox_inches="tight")
    plt.savefig(f"{PLOTS_DIR}/fig2_calibration.png", dpi=150, bbox_inches="tight")
    print("Saved fig2_calibration")


# Figure 3: Failure Prediction — Precision/Recall at thresholds
def fig3_failure_prediction():
    # Precision-recall style: if we flag episodes above uncertainty threshold
    # as 'will fail', how accurate is that prediction?# 
    from uncertainty_nav.nav_env import ENV_B

    uncs, successes, _ = _run_ensemble_episodes(ENV_B, 300)
    if uncs is None:
        print("Skipping fig3")
        return

    failures = 1.0 - successes  # 1 = failed
    thresholds = np.linspace(uncs.min(), uncs.max(), 50)

    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        predicted_fail = uncs > t
        tp = (predicted_fail & (failures == 1)).sum()
        fp = (predicted_fail & (failures == 0)).sum()
        fn = (~predicted_fail & (failures == 1)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Precision-Recall curve
    ax1.plot(recalls, precisions, "b-", linewidth=2)
    ax1.set_xlabel("Recall (fraction of failures detected)")
    ax1.set_ylabel("Precision (fraction of flags that are real failures)")
    ax1.set_title("Failure Prediction: Precision vs Recall")
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)

    # F1 vs threshold
    ax2.plot(thresholds, f1s, "r-", linewidth=2)
    best_idx = np.argmax(f1s)
    ax2.axvline(thresholds[best_idx], color="gray", linestyle="--", alpha=0.7,
                label=f"Best threshold={thresholds[best_idx]:.3f}")
    ax2.set_xlabel("Uncertainty Threshold")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Optimal Failure Detection Threshold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/fig3_failure_prediction.pdf", bbox_inches="tight")
    plt.savefig(f"{PLOTS_DIR}/fig3_failure_prediction.png", dpi=150, bbox_inches="tight")
    print("Saved fig3_failure_prediction")


# Figure 4: Ensemble Size vs Calibration Quality
def fig4_ensemble_size():
    # N vs uncertainty magnitude. N=1 has zero uncertainty (can't predict),
    # N=5+ has meaningful uncertainty
    data = _load_json(f"{RESULTS_DIR}/ablation_ensemble_size.json")
    if not data:
        print("Skipping fig4")
        return

    ns, srs, uncs = [], [], []
    for key, val in sorted(data.items(), key=lambda x: int(x[0].split("=")[1]) if "N=" in x[0] else 999):
        if not key.startswith("N=") or val.get("status"):
            continue
        ns.append(int(key.split("=")[1]))
        srs.append(val["success_rate"])
        uncs.append(val.get("mean_uncertainty", 0.0))

    if not ns:
        print("No ensemble size data")
        return

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    ax1.bar(range(len(ns)), srs, color="#2196F3", alpha=0.6, label="Success Rate")
    ax2.plot(range(len(ns)), uncs, "r-o", linewidth=2, markersize=8, label="Mean Uncertainty")

    ax1.set_xticks(range(len(ns)))
    ax1.set_xticklabels([f"N={n}" for n in ns])
    ax1.set_ylabel("Success Rate", color="#2196F3")
    ax2.set_ylabel("Mean Epistemic Uncertainty", color="r")
    ax1.set_ylim(0, 0.6)
    ax1.set_title("Ensemble Size: More Members → More Uncertainty Signal")
    ax1.annotate("N=1: no uncertainty\n(cannot predict failure)",
                 xy=(0, uncs[0]), xytext=(0.5, max(uncs)*0.5),
                 arrowprops=dict(arrowstyle="->", color="gray"),
                 fontsize=9, color="gray")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/fig4_ensemble_size.pdf", bbox_inches="tight")
    plt.savefig(f"{PLOTS_DIR}/fig4_ensemble_size.png", dpi=150, bbox_inches="tight")
    print("Saved fig4_ensemble_size")


# Figure 5: Uncertainty Timeline (single episode)
def fig5_uncertainty_timeline():
    # Uncertainty over time in one episode on Env B
    import torch
    sys.path.insert(0, "src/uncertainty_nav")
    from uncertainty_nav.models import DeepEnsemble
    from uncertainty_nav.nav_env import PartialObsNavEnv, ENV_B

    device = torch.device("cpu")
    env = PartialObsNavEnv(ENV_B, seed=42)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]

    paths = [f"checkpoints/ensemble_m{i}_policy.pt" for i in range(5)]
    existing = [p for p in paths if os.path.exists(p)]
    if not existing:
        print("Skipping fig5")
        return
    policy = DeepEnsemble.from_checkpoints(existing, obs_dim, act_dim, device=device)

    # Run multiple episodes, pick one success and one failure
    episodes = []
    for seed in range(50):
        obs, _ = env.reset(seed=seed)
        done = False
        ep_uncs = []
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, unc, _ = policy.uncertainty_driven_action(obs_t)
            obs, _, term, trunc, info = env.step(action.squeeze(0).numpy())
            done = term or trunc
            ep_uncs.append(unc.item())
        episodes.append({"uncs": ep_uncs, "success": info.get("success", False),
                         "collision": info.get("collision", False)})

    # Find one success and one failure
    success_ep = next((e for e in episodes if e["success"]), None)
    failure_ep = next((e for e in episodes if e["collision"]), None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    if success_ep:
        ax = axes[0]
        ax.plot(success_ep["uncs"], "g-", linewidth=1.5)
        ax.fill_between(range(len(success_ep["uncs"])), 0, success_ep["uncs"], alpha=0.15, color="green")
        ax.set_title("Successful Episode (low uncertainty)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Epistemic Uncertainty")
        ax.grid(alpha=0.3)

    if failure_ep:
        ax = axes[1]
        ax.plot(failure_ep["uncs"], "r-", linewidth=1.5)
        ax.fill_between(range(len(failure_ep["uncs"])), 0, failure_ep["uncs"], alpha=0.15, color="red")
        ax.set_title("Failed Episode (high uncertainty before collision)")
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)

    plt.suptitle("Uncertainty Dynamics: Success vs Failure (Env B)", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/fig5_uncertainty_timeline.pdf", bbox_inches="tight")
    plt.savefig(f"{PLOTS_DIR}/fig5_uncertainty_timeline.png", dpi=150, bbox_inches="tight")
    print("Saved fig5_uncertainty_timeline")


# Figure 6: Method Comparison (context, not main claim)
def fig6_method_comparison():
    # Bar chart: SR for all methods on Env A and B. For context only
    methods = ["vanilla", "lstm", "gru", "large_mlp", "ensemble"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    width = 0.35

    for i, env in enumerate(["A", "B"]):
        vals, errs = [], []
        for m in methods:
            data = _load_json(f"{RESULTS_DIR}/{m}_env{env}.json")
            vals.append(data["success_rate"] if data else 0.0)
            errs.append(data.get("success_rate_std", 0.0) if data else 0.0)
        offset = (i - 0.5) * width
        colors = [COLORS[m] for m in methods]
        ax.bar(x + offset, vals, width, yerr=errs, capsize=4,
               color=colors, alpha=0.6 + 0.3*i,
               label=f"Env {'A (train)' if env == 'A' else 'B (unseen)'}")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in methods], rotation=15, ha="right")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.15)
    ax.set_title("Navigation Performance (all methods degrade on unseen Env B)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/fig6_method_comparison.pdf", bbox_inches="tight")
    plt.savefig(f"{PLOTS_DIR}/fig6_method_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved fig6_method_comparison")


if __name__ == "__main__":
    fig1_ood_detection()
    fig2_calibration()
    fig3_failure_prediction()
    fig4_ensemble_size()
    fig5_uncertainty_timeline()
    fig6_method_comparison()
    print(f"\nAll plots saved to {PLOTS_DIR}/")
