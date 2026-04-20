# Renders a clean matplotlib animation showing:
# Robot trajectory color-coded by epistemic uncertainty (green→red)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib import animation
import argparse
import os
import math


from uncertainty_nav.models import PolicyNetwork, DeepEnsemble
from uncertainty_nav.baselines import VanillaMLP
from uncertainty_nav.nav_env import PartialObsNavEnv, ENV_A, ENV_B, ENV_C, ENV_D, EnvConfig

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "experiments/videos"
ENV_MAP = {"A": ENV_A, "B": ENV_B, "C": ENV_C, "D": ENV_D}

# ── Color utilities ──────────────────────────────────────────────────

def uncertainty_color(u, u_max=1.0):
    # Map uncertainty to green→yellow→red
    t = min(u / u_max, 1.0)
    if t < 0.5:
        return (2 * t, 1.0, 0.0, 0.9)       # green → yellow
    else:
        return (1.0, 2 * (1 - t), 0.0, 0.9)  # yellow → red


def load_ensemble(obs_dim, act_dim, device, n_members=5):
    paths = [f"{CHECKPOINT_DIR}/ensemble_m{i}_policy.pt" for i in range(n_members)]
    existing = [p for p in paths if os.path.exists(p)]
    if not existing:
        print("[ERROR] No ensemble checkpoints found")
        return None
    return DeepEnsemble.from_checkpoints(existing, obs_dim, act_dim, device=device)


def load_vanilla(obs_dim, act_dim, device):
    policy = VanillaMLP(obs_dim, act_dim).to(device)
    ckpt = f"{CHECKPOINT_DIR}/vanilla_policy.pt"
    if os.path.exists(ckpt):
        policy.load_state_dict(torch.load(ckpt, map_location=device))
    policy.eval()
    return policy

def run_episode_trace(env, policy, policy_type, device, threshold=0.5):
    # Run one episode, return full state trace for visualization.
    obs, _ = env.reset()
    done = False
    trace = {
        "robot_x": [], "robot_y": [], "robot_theta": [],
        "goal": (env._goal[0], env._goal[1]),
        "obstacles": [o.copy() for o in env._obstacles],
        "dynamic_obs": [],  # list of lists of (x,y) per step
        "walls": [(w[0].tolist(), w[1].tolist()) for w in env._interior_walls],
        "uncertainty": [],
        "velocity": [],
        "laser_endpoints": [],  # list of (N,2) arrays
        "actions": [],
        "success": False,
        "collision": False,
    }

    while not done:
        # Record state
        x, y, th = env._robot_pose
        trace["robot_x"].append(float(x))
        trace["robot_y"].append(float(y))
        trace["robot_theta"].append(float(th))

        # Record dynamic obstacle positions
        dyn_pos = [(d[0][0], d[0][1]) for d in env._dynamic_obstacles]
        trace["dynamic_obs"].append(dyn_pos)

        # Compute laser endpoints for visualization
        scan_raw = env._get_laser_scans()  # normalized [0,1]
        ranges = scan_raw * env.cfg.max_range
        angles = env._beam_angles_rel + th
        lx = x + ranges * np.cos(angles)
        ly = y + ranges * np.sin(angles)
        trace["laser_endpoints"].append(np.stack([lx, ly], axis=1))

        # Get action
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            if policy_type == "ensemble":
                action, unc, is_cautious = policy.uncertainty_driven_action(
                    obs_t, uncertainty_threshold=threshold)
                trace["uncertainty"].append(float(unc.item()))
            else:
                action, _ = policy.sample(obs_t)
                trace["uncertainty"].append(0.0)

        action_np = action.squeeze(0).cpu().numpy()
        v = action_np[0] * env.cfg.max_linear_vel
        trace["velocity"].append(float(abs(v)))
        trace["actions"].append(action_np.copy())

        obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

    trace["success"] = info.get("success", False)
    trace["collision"] = info.get("collision", False)
    trace["n_steps"] = len(trace["robot_x"])
    return trace

# to render static snapshot
def render_snapshot(trace, env_cfg, title="", output_path=None, step=None):
    # Render a single frame or full trajectory as a publication-quality figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=150)
    half = env_cfg.map_size / 2

    # Background
    ax.set_xlim(-half - 0.5, half + 0.5)
    ax.set_ylim(-half - 0.5, half + 0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f1a")

    # Arena boundary
    arena = patches.Rectangle((-half, -half), env_cfg.map_size, env_cfg.map_size,
                               linewidth=2, edgecolor="#444466", facecolor="none")
    ax.add_patch(arena)

    # Grid
    for g in np.arange(-half, half + 1, 1.0):
        ax.axhline(g, color="#222244", linewidth=0.3, zorder=0)
        ax.axvline(g, color="#222244", linewidth=0.3, zorder=0)

    # Interior walls
    for (x1, y1), (x2, y2) in trace["walls"]:
        ax.plot([x1, x2], [y1, y2], color="#ff6644", linewidth=3, solid_capstyle="round", zorder=3)

    # Static obstacles
    for obs_pos in trace["obstacles"]:
        circle = patches.Circle(obs_pos, env_cfg.obstacle_radius,
                                facecolor="#555577", edgecolor="#8888aa",
                                linewidth=1.5, zorder=3)
        ax.add_patch(circle)

    # Goal
    gx, gy = trace["goal"]
    goal_circle = patches.Circle((gx, gy), env_cfg.goal_radius,
                                  facecolor="#00ff8855", edgecolor="#00ff88",
                                  linewidth=2, linestyle="--", zorder=2)
    ax.add_patch(goal_circle)
    ax.plot(gx, gy, marker="*", markersize=15, color="#00ff88", zorder=5)

    # Determine which steps to draw
    n = trace["n_steps"]
    end = step if step is not None else n

    # Trajectory colored by uncertainty
    if end > 1:
        xs = trace["robot_x"][:end]
        ys = trace["robot_y"][:end]
        uncs = trace["uncertainty"][:end]
        u_max = max(max(uncs), 0.01)

        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors = [uncertainty_color(u, u_max) for u in uncs[:-1]]
        lc = LineCollection(segments, colors=colors, linewidths=2.5, zorder=4)
        ax.add_collection(lc)

    # Current robot position
    if end > 0:
        rx, ry = trace["robot_x"][end - 1], trace["robot_y"][end - 1]
        rth = trace["robot_theta"][end - 1]
        unc_now = trace["uncertainty"][end - 1] if trace["uncertainty"] else 0.0
        u_max = max(max(trace["uncertainty"]), 0.01) if trace["uncertainty"] else 1.0

        # Uncertainty halo
        halo_r = 0.2 + unc_now * 1.5
        halo_color = uncertainty_color(unc_now, u_max)
        halo = patches.Circle((rx, ry), halo_r,
                               facecolor=(*halo_color[:3], 0.15),
                               edgecolor=(*halo_color[:3], 0.6),
                               linewidth=1.5, zorder=5)
        ax.add_patch(halo)

        # Robot body
        robot = patches.Circle((rx, ry), env_cfg.collision_radius,
                                facecolor="#00aaff", edgecolor="white",
                                linewidth=1.5, zorder=6)
        ax.add_patch(robot)

        # Heading arrow
        dx = math.cos(rth) * 0.4
        dy = math.sin(rth) * 0.4
        ax.annotate("", xy=(rx + dx, ry + dy), xytext=(rx, ry),
                     arrowprops=dict(arrowstyle="->", color="white", lw=2), zorder=7)

        # Laser beams (faded)
        if end - 1 < len(trace["laser_endpoints"]):
            endpoints = trace["laser_endpoints"][end - 1]
            for i in range(0, len(endpoints), 3):  # every 3rd beam
                ax.plot([rx, endpoints[i, 0]], [ry, endpoints[i, 1]],
                        color="#ff444422", linewidth=0.5, zorder=2)

        # Dynamic obstacles at current step
        if end - 1 < len(trace["dynamic_obs"]):
            for dx_pos, dy_pos in trace["dynamic_obs"][end - 1]:
                dyn_circle = patches.Circle((dx_pos, dy_pos), env_cfg.obstacle_radius,
                                            facecolor="#ff4444", edgecolor="#ff8888",
                                            linewidth=1.5, alpha=0.8, zorder=3)
                ax.add_patch(dyn_circle)

    # Dynamic obstacle trails
    if end > 5 and trace["dynamic_obs"]:
        n_dyn = len(trace["dynamic_obs"][0]) if trace["dynamic_obs"] else 0
        for d_idx in range(n_dyn):
            dxs = [trace["dynamic_obs"][t][d_idx][0] for t in range(0, end, 3)]
            dys = [trace["dynamic_obs"][t][d_idx][1] for t in range(0, end, 3)]
            ax.plot(dxs, dys, color="#ff444444", linewidth=1, linestyle=":", zorder=2)

    # Info text
    outcome = "SUCCESS" if trace["success"] else ("COLLISION" if trace["collision"] else "TIMEOUT")
    outcome_color = "#00ff88" if trace["success"] else "#ff4444"
    info_lines = [title] if title else []
    if end > 0:
        unc_now = trace["uncertainty"][end - 1] if trace["uncertainty"] else 0.0
        vel_now = trace["velocity"][end - 1] if trace["velocity"] else 0.0
        info_lines.append(f"Step {end}/{n}  |  Unc: {unc_now:.3f}  |  Vel: {vel_now:.2f} m/s")
    if end >= n:
        info_lines.append(f"Outcome: {outcome}")
    info_text = "\n".join(info_lines)
    ax.text(-half + 0.3, half - 0.3, info_text,
            fontsize=9, color="white", fontfamily="monospace",
            verticalalignment="top", zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#00000088", edgecolor="none"))

    # Colorbar for uncertainty
    if trace["uncertainty"] and max(trace["uncertainty"]) > 0:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r,
                                    norm=plt.Normalize(0, max(trace["uncertainty"])))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Epistemic Uncertainty", color="white", fontsize=9)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    ax.set_xlabel("x (m)", color="white", fontsize=10)
    ax.set_ylabel("y (m)", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444466")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved: {output_path}")
    plt.close(fig)
    return fig


#  Video renderer 

def render_video(trace, env_cfg, title="", output_path="episode.mp4", fps=20):
    # Render full episode as MP4 video
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=120)
    half = env_cfg.map_size / 2
    n = trace["n_steps"]
    u_max = max(max(trace["uncertainty"]), 0.01) if trace["uncertainty"] else 1.0

    def init():
        ax.clear()
        return []

    def animate(frame):
        ax.clear()
        ax.set_xlim(-half - 0.5, half + 0.5)
        ax.set_ylim(-half - 0.5, half + 0.5)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a2e")

        # Arena
        arena = patches.Rectangle((-half, -half), env_cfg.map_size, env_cfg.map_size,
                                   linewidth=2, edgecolor="#444466", facecolor="none")
        ax.add_patch(arena)

        # Grid
        for g in np.arange(-half, half + 1, 2.0):
            ax.axhline(g, color="#222244", linewidth=0.3)
            ax.axvline(g, color="#222244", linewidth=0.3)

        # Walls
        for (x1, y1), (x2, y2) in trace["walls"]:
            ax.plot([x1, x2], [y1, y2], color="#ff6644", linewidth=3, solid_capstyle="round")

        # Static obstacles
        for obs_pos in trace["obstacles"]:
            ax.add_patch(patches.Circle(obs_pos, env_cfg.obstacle_radius,
                                        facecolor="#555577", edgecolor="#8888aa", linewidth=1.5))

        # Goal
        gx, gy = trace["goal"]
        ax.add_patch(patches.Circle((gx, gy), env_cfg.goal_radius,
                                     facecolor="#00ff8833", edgecolor="#00ff88",
                                     linewidth=2, linestyle="--"))
        ax.plot(gx, gy, marker="*", markersize=14, color="#00ff88")

        # Trajectory up to current frame
        end = min(frame + 1, n)
        if end > 1:
            xs = trace["robot_x"][:end]
            ys = trace["robot_y"][:end]
            uncs = trace["uncertainty"][:end]
            points = np.array([xs, ys]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            colors = [uncertainty_color(u, u_max) for u in uncs[:-1]]
            lc = LineCollection(segments, colors=colors, linewidths=2.5)
            ax.add_collection(lc)

        # Dynamic obstacles
        if frame < len(trace["dynamic_obs"]):
            for dx_pos, dy_pos in trace["dynamic_obs"][frame]:
                ax.add_patch(patches.Circle((dx_pos, dy_pos), env_cfg.obstacle_radius,
                                            facecolor="#ff4444", edgecolor="#ff8888",
                                            linewidth=1.5, alpha=0.8))
            # Trails
            n_dyn = len(trace["dynamic_obs"][0]) if trace["dynamic_obs"] else 0
            for d_idx in range(n_dyn):
                trail_end = min(frame + 1, len(trace["dynamic_obs"]))
                dxs = [trace["dynamic_obs"][t][d_idx][0] for t in range(0, trail_end, 2)]
                dys = [trace["dynamic_obs"][t][d_idx][1] for t in range(0, trail_end, 2)]
                ax.plot(dxs, dys, color="#ff444455", linewidth=1, linestyle=":")

        # Robot
        if frame < n:
            rx, ry = trace["robot_x"][frame], trace["robot_y"][frame]
            rth = trace["robot_theta"][frame]
            unc_now = trace["uncertainty"][frame]

            # Laser beams
            if frame < len(trace["laser_endpoints"]):
                endpoints = trace["laser_endpoints"][frame]
                for i in range(0, len(endpoints), 4):
                    ax.plot([rx, endpoints[i, 0]], [ry, endpoints[i, 1]],
                            color="#ff444418", linewidth=0.4)

            # Uncertainty halo
            halo_r = 0.2 + unc_now * 1.2
            hc = uncertainty_color(unc_now, u_max)
            ax.add_patch(patches.Circle((rx, ry), halo_r,
                                        facecolor=(*hc[:3], 0.12),
                                        edgecolor=(*hc[:3], 0.5), linewidth=1.5))

            # Robot body
            ax.add_patch(patches.Circle((rx, ry), env_cfg.collision_radius,
                                        facecolor="#00aaff", edgecolor="white", linewidth=1.5))

            # Heading
            dx = math.cos(rth) * 0.35
            dy = math.sin(rth) * 0.35
            ax.annotate("", xy=(rx + dx, ry + dy), xytext=(rx, ry),
                         arrowprops=dict(arrowstyle="->", color="white", lw=1.8))

        # HUD
        unc_now = trace["uncertainty"][min(frame, n - 1)]
        vel_now = trace["velocity"][min(frame, n - 1)]
        step_text = f"Step {min(frame + 1, n)}/{n}"
        unc_text = f"Uncertainty: {unc_now:.3f}"
        vel_text = f"Velocity: {vel_now:.2f} m/s"
        if frame >= n - 1:
            outcome = "SUCCESS" if trace["success"] else ("COLLISION" if trace["collision"] else "TIMEOUT")
            step_text += f"  [{outcome}]"
        hud = f"{title}\n{step_text}\n{unc_text}\n{vel_text}"
        ax.text(-half + 0.2, half - 0.2, hud,
                fontsize=9, color="white", fontfamily="monospace",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#00000099", edgecolor="none"))

        ax.set_xlabel("x (m)", color="white", fontsize=9)
        ax.set_ylabel("y (m)", color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#444466")

        return []

    # Add extra frames at the end to hold the final state
    total_frames = n + fps * 2  # hold last frame for 2 seconds
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=total_frames, interval=1000 // fps, blit=False)

    fig.patch.set_facecolor("#0f0f1a")
    writer = animation.FFMpegWriter(fps=fps, bitrate=3000,
                                     extra_args=["-pix_fmt", "yuv420p"])
    anim.save(output_path, writer=writer, facecolor=fig.get_facecolor())
    print(f"  Video saved: {output_path}")
    plt.close(fig)


#  Side-by-side comparison renderer

def render_comparison(trace_ens, trace_base, env_cfg, title="", output_path=None):
    # Render ensemble vs baseline side by side as a static figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    fig.patch.set_facecolor("#0f0f1a")

    for ax, trace, label in [(ax1, trace_ens, "Ensemble (Ours)"),
                              (ax2, trace_base, "Vanilla MLP")]:
        half = env_cfg.map_size / 2
        ax.set_xlim(-half - 0.5, half + 0.5)
        ax.set_ylim(-half - 0.5, half + 0.5)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a2e")

        # Arena + grid
        ax.add_patch(patches.Rectangle((-half, -half), env_cfg.map_size, env_cfg.map_size,
                                        linewidth=2, edgecolor="#444466", facecolor="none"))
        for g in np.arange(-half, half + 1, 2.0):
            ax.axhline(g, color="#222244", linewidth=0.3)
            ax.axvline(g, color="#222244", linewidth=0.3)

        # Walls
        for (x1, y1), (x2, y2) in trace["walls"]:
            ax.plot([x1, x2], [y1, y2], color="#ff6644", linewidth=3, solid_capstyle="round")

        # Obstacles
        for obs_pos in trace["obstacles"]:
            ax.add_patch(patches.Circle(obs_pos, env_cfg.obstacle_radius,
                                        facecolor="#555577", edgecolor="#8888aa", linewidth=1.5))

        # Goal
        gx, gy = trace["goal"]
        ax.add_patch(patches.Circle((gx, gy), env_cfg.goal_radius,
                                     facecolor="#00ff8833", edgecolor="#00ff88", linewidth=2, linestyle="--"))
        ax.plot(gx, gy, marker="*", markersize=14, color="#00ff88")

        # Trajectory
        n = trace["n_steps"]
        if n > 1:
            xs, ys = trace["robot_x"], trace["robot_y"]
            uncs = trace["uncertainty"]
            u_max = max(max(uncs), 0.01)
            points = np.array([xs, ys]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            colors = [uncertainty_color(u, u_max) for u in uncs[:-1]]
            lc = LineCollection(segments, colors=colors, linewidths=2.5)
            ax.add_collection(lc)

        # Start and end markers
        ax.plot(trace["robot_x"][0], trace["robot_y"][0], "o", color="#00aaff", markersize=8, zorder=6)
        ax.plot(trace["robot_x"][-1], trace["robot_y"][-1], "s", color="white", markersize=8, zorder=6)

        # Dynamic obstacle trails
        if trace["dynamic_obs"]:
            n_dyn = len(trace["dynamic_obs"][0])
            for d_idx in range(n_dyn):
                dxs = [trace["dynamic_obs"][t][d_idx][0] for t in range(0, n, 3)]
                dys = [trace["dynamic_obs"][t][d_idx][1] for t in range(0, n, 3)]
                ax.plot(dxs, dys, color="#ff444466", linewidth=1.5, linestyle=":")

        # Info
        outcome = "SUCCESS" if trace["success"] else ("COLLISION" if trace["collision"] else "TIMEOUT")
        oc = "#00ff88" if trace["success"] else "#ff4444"
        mean_unc = np.mean(trace["uncertainty"]) if trace["uncertainty"] else 0.0
        info = f"{label}\n{outcome} ({n} steps)\nMean unc: {mean_unc:.3f}"
        ax.text(-half + 0.3, half - 0.3, info,
                fontsize=10, color="white", fontfamily="monospace",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#00000099", edgecolor="none"))
        ax.set_title(label, color="white", fontsize=13, fontweight="bold", pad=10)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#444466")

    plt.suptitle(title, color="white", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved: {output_path}")
    plt.close(fig)


# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D visualization of uncertainty-aware navigation")
    parser.add_argument("--env", default="B", choices=["A", "B", "C", "D"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-video", action="store_true", help="Skip video, only snapshots")
    parser.add_argument("--snapshots", action="store_true", help="Save key frame snapshots")
    parser.add_argument("--compare", default=None, help="Compare with baseline (vanilla)")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cpu")
    env_cfg = ENV_MAP[args.env]
    env = PartialObsNavEnv(env_cfg, seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run ensemble episode
    print(f"\n{'='*50}")
    print(f"Running ensemble on Env {args.env} (seed={args.seed})")
    print(f"{'='*50}")
    ensemble = load_ensemble(obs_dim, act_dim, device)
    if ensemble is None:
        exit(1)
    env_ens = PartialObsNavEnv(env_cfg, seed=args.seed)
    trace_ens = run_episode_trace(env_ens, ensemble, "ensemble", device)
    outcome = "SUCCESS" if trace_ens["success"] else ("COLLISION" if trace_ens["collision"] else "TIMEOUT")
    print(f"  Outcome: {outcome} in {trace_ens['n_steps']} steps")
    print(f"  Mean uncertainty: {np.mean(trace_ens['uncertainty']):.4f}")

    title = f"Epistemic Ensemble — Env {args.env}"

    # Video
    if not args.no_video:
        print("\nRendering video...")
        render_video(trace_ens, env_cfg, title=title,
                     output_path=f"{OUTPUT_DIR}/ensemble_env{args.env}_seed{args.seed}.mp4",
                     fps=args.fps)

    # Snapshots
    if args.snapshots or args.no_video:
        print("\nRendering snapshots...")
        n = trace_ens["n_steps"]
        for frac, label in [(0.0, "start"), (0.25, "quarter"), (0.5, "half"),
                             (0.75, "three_quarter"), (1.0, "end")]:
            step = max(1, int(frac * n))
            render_snapshot(trace_ens, env_cfg, title=f"{title} (step {step}/{n})",
                           output_path=f"{OUTPUT_DIR}/snapshot_env{args.env}_{label}.png",
                           step=step)

    # Comparison
    if args.compare:
        print(f"\nRunning {args.compare} baseline for comparison...")
        if args.compare == "vanilla":
            baseline = load_vanilla(obs_dim, act_dim, device)
        else:
            print(f"  Unknown baseline: {args.compare}")
            exit(1)
        env_base = PartialObsNavEnv(env_cfg, seed=args.seed)
        trace_base = run_episode_trace(env_base, baseline, args.compare, device)
        outcome_b = "SUCCESS" if trace_base["success"] else ("COLLISION" if trace_base["collision"] else "TIMEOUT")
        print(f"  Baseline outcome: {outcome_b} in {trace_base['n_steps']} steps")

        render_comparison(trace_ens, trace_base, env_cfg,
                          title=f"Env {args.env}: Ensemble vs {args.compare.title()}",
                          output_path=f"{OUTPUT_DIR}/comparison_env{args.env}_seed{args.seed}.png")

    print(f"\nAll outputs in {OUTPUT_DIR}/")
    print("Done.")

