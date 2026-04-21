#!/usr/bin/env python3
# Comprehensive evaluation across all environments (A, B, C, D)

import torch
import numpy as np
import json
import argparse
import os
from collections import defaultdict

from uncertainty_nav.models import PolicyNetwork, DeepEnsemble
from uncertainty_nav.baselines import VanillaMLP, RecurrentPolicy, LargeMLPPolicy
from uncertainty_nav.mc_dropout import MCDropoutPolicy
from uncertainty_nav.nav_env import (
    PartialObsNavEnv, ENV_A, ENV_B, ENV_C, ENV_D, EnvConfig
)

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "experiments/results"

ENV_MAP = {"A": ENV_A, "B": ENV_B, "C": ENV_C, "D": ENV_D}


def load_policy(policy_type, obs_dim, act_dim, device, n_members=5):
    if policy_type == "ensemble":
        paths = [f"{CHECKPOINT_DIR}/ensemble_m{i}_policy.pt" for i in range(n_members)]
        existing = [p for p in paths if os.path.exists(p)]
        if not existing:
            print(f"[WARNING] No ensemble member checkpoints found")
            return None
        policy = DeepEnsemble.from_checkpoints(existing, obs_dim, act_dim, device=device)
        print(f"  Loaded ensemble with {len(existing)} members")
        return policy
    elif policy_type in ("lstm", "gru"):
        policy = RecurrentPolicy(obs_dim, act_dim, rnn_type=policy_type).to(device)
    elif policy_type == "large_mlp":
        policy = LargeMLPPolicy(obs_dim, act_dim).to(device)
    elif policy_type == "mc_dropout":
        policy = MCDropoutPolicy(obs_dim, act_dim).to(device)
    else:
        policy = VanillaMLP(obs_dim, act_dim).to(device)

    ckpt = f"{CHECKPOINT_DIR}/{policy_type}_policy.pt"
    if os.path.exists(ckpt):
        policy.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        print(f"[WARNING] Checkpoint not found: {ckpt}")
        return None
    policy.eval()
    return policy


def run_episode(env, policy, policy_type, device, uncertainty_threshold=0.5):
    obs, _ = env.reset()
    hidden = policy.init_hidden() if policy_type in ("lstm", "gru") else None
    done = False
    total_reward = 0.0
    uncertainties = []
    cautious_steps = 0
    steps = 0

    while not done:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            if policy_type == "ensemble":
                action, unc, is_cautious = policy.uncertainty_driven_action(
                    obs_t, uncertainty_threshold=uncertainty_threshold
                )
                uncertainties.append(unc.item())
                cautious_steps += int(is_cautious)
            elif policy_type == "mc_dropout":
                action, unc, is_cautious = policy.uncertainty_driven_action(
                    obs_t, uncertainty_threshold=uncertainty_threshold,
                    n_samples=20
                )
                uncertainties.append(unc.item())
                cautious_steps += int(is_cautious)
            elif policy_type in ("lstm", "gru"):
                action, _, hidden = policy.sample(obs_t, hidden)
            else:
                action, _ = policy.sample(obs_t)

        action_np = action.squeeze(0).cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    success = info.get("success", False)
    collision = info.get("collision", False)
    timeout = (not success) and (not collision)  # Pillar 3d: separate timeouts
    spl = env.compute_spl() if success else 0.0

    return {
        "success": success,
        "collision": collision,
        "timeout": timeout,
        "total_reward": total_reward,
        "spl": spl,
        "steps": steps,
        "path_length": info.get("path_length", 0.0),
        "optimal_path_length": info.get("optimal_path_length", 0.0),
        "mean_uncertainty": float(np.mean(uncertainties)) if uncertainties else 0.0,
        "std_uncertainty": float(np.std(uncertainties)) if uncertainties else 0.0,
        "cautious_step_ratio": cautious_steps / max(steps, 1),
    }


def compute_ece(uncertainties, failures, n_bins=10):
    """Expected Calibration Error: how well does uncertainty predict failure?"""
    uncertainties = np.array(uncertainties)
    failures = np.array(failures, dtype=float)
    bin_edges = np.linspace(0, uncertainties.max() + 1e-8, n_bins + 1)
    ece = 0.0
    bin_data = []
    for i in range(n_bins):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_failure_rate = failures[mask].mean()
        bin_mean_unc = uncertainties[mask].mean()
        bin_size = mask.sum()
        # Normalize uncertainty to [0,1] for ECE computation
        norm_unc = bin_mean_unc / (uncertainties.max() + 1e-8)
        ece += (bin_size / len(uncertainties)) * abs(bin_failure_rate - norm_unc)
        bin_data.append({
            "bin": i,
            "n": int(bin_size),
            "mean_unc": float(bin_mean_unc),
            "failure_rate": float(bin_failure_rate),
        })
    return float(ece), bin_data


def compute_auroc(uncertainties, failures):
    """AUROC for failure prediction using uncertainty as the score."""
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        uncertainties = np.array(uncertainties)
        failures = np.array(failures, dtype=float)
        if len(np.unique(failures)) < 2:
            return 0.0, []
        auroc = roc_auc_score(failures, uncertainties)
        fpr, tpr, thresholds = roc_curve(failures, uncertainties)
        roc_data = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr[::5], tpr[::5])]
        return float(auroc), roc_data
    except ImportError:
        # Fallback: manual AUROC via Mann-Whitney U statistic
        uncertainties = np.array(uncertainties)
        failures = np.array(failures, dtype=float)
        pos = uncertainties[failures == 1]
        neg = uncertainties[failures == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.0, []
        auroc = np.mean([float(p > n) + 0.5 * float(p == n)
                         for p in pos for n in neg])
        return float(auroc), []


def compute_routing_table(episodes):
    """Pillar 1: Selective deployment analysis.
    Route episodes by uncertainty quartile to compute autonomous SR
    at different human burden levels."""
    uncs = np.array([e["mean_uncertainty"] for e in episodes])
    successes = np.array([e["success"] for e in episodes], dtype=float)
    quartiles = np.percentile(uncs, [25, 50, 75])
    bins = np.digitize(uncs, quartiles)  # 0=Q0, 1=Q1, 2=Q2, 3=Q3

    routing = []
    # Strategy 1: No routing (all autonomous)
    routing.append({
        "strategy": "No routing (all autonomous)",
        "autonomous_frac": 1.0,
        "autonomous_sr": float(successes.mean()),
        "human_burden": 0.0,
    })
    # Strategy 2: Route Q3 to human
    mask = bins <= 2
    routing.append({
        "strategy": "Route Q3 to human",
        "autonomous_frac": float(mask.mean()),
        "autonomous_sr": float(successes[mask].mean()) if mask.sum() > 0 else 0.0,
        "human_burden": float((~mask).mean()),
    })
    # Strategy 3: Route Q2+Q3 to human
    mask = bins <= 1
    routing.append({
        "strategy": "Route Q2+Q3 to human",
        "autonomous_frac": float(mask.mean()),
        "autonomous_sr": float(successes[mask].mean()) if mask.sum() > 0 else 0.0,
        "human_burden": float((~mask).mean()),
    })
    # Strategy 4: Route only Q0 autonomous
    mask = bins == 0
    routing.append({
        "strategy": "Only Q0 autonomous",
        "autonomous_frac": float(mask.mean()),
        "autonomous_sr": float(successes[mask].mean()) if mask.sum() > 0 else 0.0,
        "human_burden": float((~mask).mean()),
    })
    return routing


def evaluate_single(policy_type, env_name, device, n_episodes=200, n_seeds=5):
    """Evaluate a single policy on a single environment with full metrics."""
    env_cfg = ENV_MAP[env_name]
    obs_dim = PartialObsNavEnv(env_cfg).observation_space.shape[0]
    act_dim = PartialObsNavEnv(env_cfg).action_space.shape[0]
    policy = load_policy(policy_type, obs_dim, act_dim, device)
    if policy is None:
        return None

    seed_metrics = defaultdict(list)
    all_episodes = []
    eps_per_seed = n_episodes // n_seeds

    for seed in range(n_seeds):
        env = PartialObsNavEnv(env_cfg, seed=seed + 100)
        episodes = [run_episode(env, policy, policy_type, device)
                    for _ in range(eps_per_seed)]
        all_episodes.extend(episodes)
        seed_metrics["success_rate"].append(np.mean([e["success"] for e in episodes]))
        seed_metrics["collision_rate"].append(np.mean([e["collision"] for e in episodes]))
        seed_metrics["timeout_rate"].append(np.mean([e["timeout"] for e in episodes]))
        seed_metrics["mean_spl"].append(np.mean([e["spl"] for e in episodes]))
        seed_metrics["mean_reward"].append(np.mean([e["total_reward"] for e in episodes]))

    metrics = {
        "policy": policy_type,
        "env": env_name,
        "n_episodes": n_episodes,
        "n_seeds": n_seeds,
        # Core metrics
        "success_rate": float(np.mean(seed_metrics["success_rate"])),
        "success_rate_std": float(np.std(seed_metrics["success_rate"])),
        "collision_rate": float(np.mean(seed_metrics["collision_rate"])),
        "collision_rate_std": float(np.std(seed_metrics["collision_rate"])),
        "timeout_rate": float(np.mean(seed_metrics["timeout_rate"])),
        "timeout_rate_std": float(np.std(seed_metrics["timeout_rate"])),
        "mean_spl": float(np.mean(seed_metrics["mean_spl"])),
        "mean_spl_std": float(np.std(seed_metrics["mean_spl"])),
        "mean_reward": float(np.mean(seed_metrics["mean_reward"])),
        "mean_path_length": float(np.mean([e["path_length"] for e in all_episodes])),
        "mean_steps": float(np.mean([e["steps"] for e in all_episodes])),
        # Uncertainty metrics
        "mean_uncertainty": float(np.mean([e["mean_uncertainty"] for e in all_episodes])),
        "std_uncertainty": float(np.mean([e["std_uncertainty"] for e in all_episodes])),
        "mean_cautious_ratio": float(np.mean([e["cautious_step_ratio"] for e in all_episodes])),
    }

    # Ensemble-specific advanced metrics
    if policy_type in ("ensemble", "mc_dropout"):
        uncs = np.array([e["mean_uncertainty"] for e in all_episodes])
        successes = np.array([e["success"] for e in all_episodes], dtype=float)
        failures = 1.0 - successes

        # Correlation
        corr = float(np.corrcoef(uncs, successes)[0, 1]) if uncs.std() > 1e-8 else 0.0
        metrics["uncertainty_success_correlation"] = corr

        # Calibration quartiles
        quartiles = np.percentile(uncs, [25, 50, 75])
        bins = np.digitize(uncs, quartiles)
        calibration = {}
        for b in range(4):
            mask = bins == b
            if mask.sum() > 0:
                calibration[f"q{b}_sr"] = float(successes[mask].mean())
                calibration[f"q{b}_cr"] = float(np.mean([e["collision"] for e, m in zip(all_episodes, mask) if m]))
                calibration[f"q{b}_timeout"] = float(np.mean([e["timeout"] for e, m in zip(all_episodes, mask) if m]))
                calibration[f"q{b}_mean_unc"] = float(uncs[mask].mean())
                calibration[f"q{b}_n"] = int(mask.sum())
        metrics["uncertainty_calibration"] = calibration

        # ECE (Pillar 3b)
        ece, ece_bins = compute_ece(uncs, failures)
        metrics["ece"] = ece
        metrics["ece_bins"] = ece_bins

        # AUROC (Pillar 3b)
        auroc, roc_data = compute_auroc(uncs, failures)
        metrics["auroc_failure"] = auroc
        metrics["roc_curve"] = roc_data

        # Routing table (Pillar 1)
        routing = compute_routing_table(all_episodes)
        metrics["routing_table"] = routing

    # Print summary
    print(f"\n{'='*60}")
    print(f"Policy: {policy_type} | Env: {env_name} | Episodes: {n_episodes}")
    print(f"  SR:        {metrics['success_rate']:.3f} +/- {metrics['success_rate_std']:.3f}")
    print(f"  SPL:       {metrics['mean_spl']:.3f} +/- {metrics['mean_spl_std']:.3f}")
    print(f"  Collision: {metrics['collision_rate']:.3f} +/- {metrics['collision_rate_std']:.3f}")
    print(f"  Timeout:   {metrics['timeout_rate']:.3f} +/- {metrics['timeout_rate_std']:.3f}")
    print(f"  Reward:    {metrics['mean_reward']:.2f}")
    if policy_type in ("ensemble", "mc_dropout"):
        print(f"  Unc:       {metrics['mean_uncertainty']:.4f}")
        print(f"  Corr:      {metrics.get('uncertainty_success_correlation', 0):.3f}")
        print(f"  ECE:       {metrics.get('ece', 0):.4f}")
        print(f"  AUROC:     {metrics.get('auroc_failure', 0):.4f}")
        print(f"  Routing:")
        for r in metrics.get("routing_table", []):
            print(f"    {r['strategy']}: auto={r['autonomous_frac']:.0%} "
                  f"SR={r['autonomous_sr']:.3f} human={r['human_burden']:.0%}")
    print(f"{'='*60}")

    return metrics


def evaluate_env_e_sweep(device, n_episodes=100, n_seeds=5):
    """Env E: Noise sweep on Env B layout (Pillar 2).
    Tests ensemble uncertainty tracking across a continuous difficulty spectrum."""
    noise_levels = [0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.30]
    results = {}

    for policy_type in ["ensemble", "vanilla", "large_mlp"]:
        policy_results = {}
        for sigma in noise_levels:
            cfg = EnvConfig(
                map_size=12.0,
                laser_noise_std=sigma,
                occlusion_prob=0.20,
                fov_deg=120.0,
                n_static_obstacles=10,
                dropout_prob=0.08,
                max_steps=600,
                interior_walls=(
                    (-2.0, -4.0, -2.0, 2.0),
                    (2.0, -2.0, 2.0, 4.0),
                ),
                n_dynamic_obstacles=2,
                dynamic_speed=0.10,
            )
            obs_dim = PartialObsNavEnv(cfg).observation_space.shape[0]
            act_dim = PartialObsNavEnv(cfg).action_space.shape[0]
            policy = load_policy(policy_type, obs_dim, act_dim, device)
            if policy is None:
                continue

            all_eps = []
            eps_per_seed = n_episodes // n_seeds
            for seed in range(n_seeds):
                env = PartialObsNavEnv(cfg, seed=seed + 200)
                for _ in range(eps_per_seed):
                    ep = run_episode(env, policy, policy_type, device)
                    all_eps.append(ep)

            sr = float(np.mean([e["success"] for e in all_eps]))
            cr = float(np.mean([e["collision"] for e in all_eps]))
            unc = float(np.mean([e["mean_uncertainty"] for e in all_eps]))
            policy_results[str(sigma)] = {
                "sr": sr, "cr": cr, "mean_uncertainty": unc,
                "n_episodes": len(all_eps),
            }
            print(f"  Env E [{policy_type}] sigma={sigma:.2f}: SR={sr:.3f} CR={cr:.3f} unc={unc:.4f}")

        results[policy_type] = policy_results

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/env_e_noise_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR}/env_e_noise_sweep.json")
    return results


def evaluate_ensemble_size_auroc(device, n_episodes=500, n_seeds=5):
    """Pillar 3c: AUROC by ensemble size to justify N=5 over N=2."""
    env_cfg = ENV_B
    obs_dim = PartialObsNavEnv(env_cfg).observation_space.shape[0]
    act_dim = PartialObsNavEnv(env_cfg).action_space.shape[0]
    results = {}

    for n in [1, 2, 3, 5, 10]:
        paths = [f"{CHECKPOINT_DIR}/ensemble_m{i}_policy.pt" for i in range(n)]
        existing = [p for p in paths if os.path.exists(p)]
        if not existing:
            print(f"  N={n}: skipped (no checkpoints)")
            continue
        ens = DeepEnsemble.from_checkpoints(existing, obs_dim, act_dim, device=device)

        all_eps = []
        eps_per_seed = n_episodes // n_seeds
        for seed in range(n_seeds):
            env = PartialObsNavEnv(env_cfg, seed=seed + 300)
            for _ in range(eps_per_seed):
                ep = run_episode(env, ens, "ensemble", device)
                all_eps.append(ep)

        uncs = np.array([e["mean_uncertainty"] for e in all_eps])
        failures = np.array([1.0 - e["success"] for e in all_eps])
        successes = 1.0 - failures

        auroc, _ = compute_auroc(uncs, failures)
        ece, _ = compute_ece(uncs, failures)
        sr = float(successes.mean())
        cr = float(np.mean([e["collision"] for e in all_eps]))
        tr = float(np.mean([e["timeout"] for e in all_eps]))

        results[f"N={n}"] = {
            "success_rate": sr,
            "collision_rate": cr,
            "timeout_rate": tr,
            "mean_uncertainty": float(uncs.mean()),
            "auroc": auroc,
            "ece": ece,
            "n_episodes": len(all_eps),
        }
        print(f"  N={n}: SR={sr:.3f} CR={cr:.3f} TO={tr:.3f} "
              f"AUROC={auroc:.4f} ECE={ece:.4f} unc={uncs.mean():.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/ensemble_size_auroc.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR}/ensemble_size_auroc.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive evaluation for ICRA revision")
    parser.add_argument("--policy", default="ensemble",
                        choices=["ensemble", "vanilla", "lstm", "gru", "large_mlp", "mc_dropout"])
    parser.add_argument("--env", default="B", choices=["A", "B", "C", "D"])
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--all", action="store_true", help="Run all policies on all envs")
    parser.add_argument("--env_e", action="store_true", help="Run Env E noise sweep")
    parser.add_argument("--auroc_ablation", action="store_true", help="AUROC by ensemble size")
    args = parser.parse_args()

    device = torch.device("cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.all:
        print("=" * 60)
        print(" COMPREHENSIVE EVALUATION: All policies x All environments")
        print("=" * 60)
        policies = ["ensemble", "vanilla", "lstm", "gru", "large_mlp", "mc_dropout"]
        envs = ["A", "B", "C", "D"]
        for env_name in envs:
            for pol in policies:
                print(f"\n>>> Evaluating {pol} on Env {env_name}...")
                m = evaluate_single(pol, env_name, device, args.n_episodes, args.n_seeds)
                if m:
                    path = f"{OUTPUT_DIR}/{pol}_env{env_name}.json"
                    with open(path, "w") as f:
                        json.dump(m, f, indent=2)
                    print(f"  Saved: {path}")

    elif args.env_e:
        print("=" * 60)
        print(" ENV E: Noise sweep on Env B layout")
        print("=" * 60)
        evaluate_env_e_sweep(device)

    elif args.auroc_ablation:
        print("=" * 60)
        print(" AUROC by ensemble size (Pillar 3c)")
        print("=" * 60)
        evaluate_ensemble_size_auroc(device)

    else:
        m = evaluate_single(args.policy, args.env, device, args.n_episodes, args.n_seeds)
        if m:
            path = f"{OUTPUT_DIR}/{args.policy}_env{args.env}.json"
            with open(path, "w") as f:
                json.dump(m, f, indent=2)
            print(f"Saved: {path}")

    print("\nDone.")
