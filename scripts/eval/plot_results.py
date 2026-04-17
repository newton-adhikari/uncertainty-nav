# Paper figures for uncertainty calibration and failure prediction

import os
import matplotlib.pyplot as plt

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

def fig1_ood_detection():
    # Histogram: uncertainty distribution on Env A vs Env B.
    # Shows the ensemble detects distribution shift

    # import both envs
    from uncertainty_nav.nav_env import ENV_A, ENV_B



if __name__ == "__main__":
    fig1_ood_detection()
