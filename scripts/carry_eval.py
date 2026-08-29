"""Headless rollout eval for the Carry task (roadmap Phase 2).

Answers the questions a reward curve cannot: does the trained policy actually
WALK while carrying, does it stay upright, and how hard does it shake the toy?
Runs on CPU, so it needs no GPU.

    uv run scripts/carry_eval.py --checkpoint /path/model_3999.pt --num-envs 32

Reports, per checkpoint:
  * survival     — mean episode length and the share of episodes ending in a fall
  * tracking     — |achieved - commanded| for planar velocity and yaw rate,
                   split into WALKING and STANDING commands, because the velocity
                   recipe deliberately commands ~25% of envs to stand still and
                   averaging the two together hides both
  * payload      — carried toy acceleration and the grip force the beak must
                   supply (the number roadmap Phase 6 needs)
  * weld         — how far the toy drifts from the pose it is welded at, which is
                   the check that "carrying" is still physically true at the end
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch


def evaluate(
    checkpoint: Path,
    task_id: str = "Mjlab-Carry-Flat-MicroDuck",
    num_envs: int = 32,
    steps: int = 400,
    device: str = "cpu",
    seed: int = 0,
    play: bool = True,
    push_interval: tuple[float, float] | None = None,
) -> dict:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import (
        load_env_cfg,
        load_rl_cfg,
        load_runner_cls,
    )
    from mjlab.utils.lab_api.math import quat_apply_inverse

    from mjlab_microduck.tasks.microduck_carry_env_cfg import CARRY_OFFSET_IN_HEAD

    env_cfg = load_env_cfg(task_id, play=play)
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed

    # The inherited PLAY config pushes every 0.5-1.0 s, six times more often than
    # the 3-6 s the policy was trained under. That is a robustness battery, not a
    # like-for-like eval, and reading it as "the policy fails" is exactly the
    # mistake AGENTS.md warns about (measure before theorizing). Allow both.
    if push_interval is not None and "push_robot" in env_cfg.events:
        env_cfg.events["push_robot"].interval_range_s = tuple(push_interval)

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True,
                map_location=device)
    policy = runner.get_inference_policy(device=device)

    robot = base_env.scene["robot"]
    toy = base_env.scene["toy"]
    head_idx = robot.body_names.index("jaw_soft")
    want = torch.tensor(CARRY_OFFSET_IN_HEAD, device=device).unsqueeze(0)

    err_xy_walk, err_yaw_walk, err_xy_stand = [], [], []
    speeds, accels, forces, weld_errs = [], [], [], []
    falls = 0
    ends = 0
    ep_lengths = []
    # Track episode length ourselves: env.step resets terminated envs before it
    # returns, zeroing episode_length_buf, so reading it after the fact is always 0.
    alive = torch.zeros(num_envs, device=device)

    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(steps):
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

            cmd = base_env.command_manager.get_command("twist")
            lin = robot.data.root_link_lin_vel_b[:, :2]
            yaw = robot.data.root_link_ang_vel_b[:, 2]

            walking = cmd[:, :3].abs().sum(dim=-1) > 1e-6
            e_xy = torch.linalg.norm(lin - cmd[:, :2], dim=-1)
            e_yaw = (yaw - cmd[:, 2]).abs()
            if bool(walking.any()):
                err_xy_walk.append(e_xy[walking].mean().item())
                err_yaw_walk.append(e_yaw[walking].mean().item())
                speeds.append(torch.linalg.norm(lin[walking], dim=-1).mean().item())
            if bool((~walking).any()):
                err_xy_stand.append(e_xy[~walking].mean().item())

            from mjlab_microduck.tasks.mdp import (
                carried_toy_accel,
                carried_toy_grip_force,
            )

            accels.append(carried_toy_accel(base_env).mean().item())
            forces.append(carried_toy_grip_force(base_env).max().item())

            head_bid = int(robot.indexing.body_ids[head_idx])
            rel = toy.data.root_link_pos_w - base_env.sim.data.xpos[:, head_bid]
            in_head = quat_apply_inverse(base_env.sim.data.xquat[:, head_bid], rel)
            weld_errs.append(torch.linalg.norm(in_head - want, dim=-1).max().item())

            alive += 1.0
            done_mask = dones.bool()
            n_done = int(done_mask.sum())
            if n_done:
                ends += n_done
                term = base_env.termination_manager
                if "fell_over" in term.active_terms:
                    falls += int(term.get_term("fell_over")[done_mask].sum())
                ep_lengths.append(alive[done_mask].mean().item())
                alive[done_mask] = 0.0

    env.close()
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    return {
        "checkpoint": checkpoint.name,
        "episodes_ended": ends,
        "fall_rate": (falls / ends) if ends else 0.0,
        "mean_ep_len_at_end": mean(ep_lengths),
        "err_vel_xy_walking": mean(err_xy_walk),
        "err_yaw_walking": mean(err_yaw_walk),
        "err_vel_xy_standing": mean(err_xy_stand),
        "mean_speed_walking": mean(speeds),
        "toy_accel_mean": mean(accels),
        "toy_grip_force_max": max(forces) if forces else float("nan"),
        "weld_drift_max_mm": (max(weld_errs) * 1e3) if weld_errs else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-play", action="store_true",
                    help="Use the TRAIN config instead of the play config.")
    ap.add_argument("--push-interval", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="Override push interval in seconds (train uses 3 6).")
    args = ap.parse_args()

    results = []
    for ckpt in args.checkpoint:
        print(f"\n--- evaluating {ckpt.name} ---", flush=True)
        results.append(
            evaluate(ckpt, num_envs=args.num_envs, steps=args.steps,
                     device=args.device, play=not args.no_play,
                     push_interval=args.push_interval)
        )

    print("\n" + "=" * 118)
    hdr = (f"{'checkpoint':>16}{'falls':>8}{'eplen':>8}{'err_xy':>9}{'err_yaw':>9}"
           f"{'stand_err':>11}{'speed':>8}{'toy_acc':>9}{'grip_N':>8}{'weld_mm':>9}")
    print(hdr)
    print("-" * 118)
    for r in results:
        print(f"{r['checkpoint']:>16}{r['fall_rate']:8.2%}{r['mean_ep_len_at_end']:8.0f}"
              f"{r['err_vel_xy_walking']:9.3f}{r['err_yaw_walking']:9.3f}"
              f"{r['err_vel_xy_standing']:11.3f}{r['mean_speed_walking']:8.3f}"
              f"{r['toy_accel_mean']:9.2f}{r['toy_grip_force_max']:8.3f}"
              f"{r['weld_drift_max_mm']:9.3f}")
    print("=" * 118)


if __name__ == "__main__":
    main()
