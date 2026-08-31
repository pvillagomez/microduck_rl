"""Measure the Carry task's geometry: where the toy rides, and what it does to the CoM.

CPU only, no GPU needed. Run this after ANY re-export of the robot model — the
constants in ``microduck_carry_env_cfg.py`` are measured off the head collision
mesh, and a re-export can silently move them (AGENTS.md: never carry measured
heights across model revisions).

    uv run scripts/carry_sweep.py

Reports:
  1. The head frame and where ``mouth_tip`` sits in it.
  2. A penetration sweep: how far below ``mouth_tip`` the toy must hang before it
     stops interpenetrating the head collision mesh. The first clear offset is
     CARRY_OFFSET_IN_HEAD — a welded toy that overlaps the beak fights a contact
     on every single step.
  3. The whole-body CoM shift the carried toy causes, per payload mass. This is
     the quantity the roadmap flags as the real Phase 2 difficulty: walking turns
     a static CoM offset into a dynamic disturbance.
"""

from copy import deepcopy

import mujoco
import numpy as np

from mjlab.scene import Scene, SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_GRASP_LIFT_ROBOT_CFG,
    MICRODUCK_TOY_CFG,
)
from mjlab_microduck.tasks.mdp import make_grasp_weld_spec_fn

# Sole length the payload budget is expressed against (Phase 1, payload_sweep.py).
SOLE_LENGTH_M = 0.046


def build():
    scene = Scene(
        SceneCfg(
            num_envs=1,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={
                "robot": deepcopy(MICRODUCK_GRASP_LIFT_ROBOT_CFG),
                "toy": deepcopy(MICRODUCK_TOY_CFG),
            },
            spec_fn=make_grasp_weld_spec_fn(),
        ),
        device="cpu",
    )
    model = scene.compile()
    return model, mujoco.MjData(model)


def main() -> None:
    model, data = build()

    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/jaw_soft")
    toy_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "toy/toy")
    toy_g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "toy/toy_geom")
    mouth = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "robot/mouth_tip")
    qadr = model.jnt_qposadr[model.body_jntadr[toy_b]]

    def reset_home():
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)

    def head_frame():
        return data.xpos[head].copy(), data.xmat[head].reshape(3, 3).copy()

    def place(offset_in_head):
        """Place the toy at an offset in the CURRENT head frame; verify it landed."""
        x_head, R_head = head_frame()
        target = x_head + R_head @ np.asarray(offset_in_head)
        data.qpos[qadr : qadr + 3] = target
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R_head.flatten())
        data.qpos[qadr + 3 : qadr + 7] = q
        mujoco.mj_forward(model, data)
        # Guard against reading a stale head frame — the first version of this
        # script cached x_head across mj_setConst calls and silently placed the toy
        # 145 mm from where it reported.
        err = float(np.linalg.norm(data.xpos[toy_b] - target))
        assert err < 1e-9, f"placement drifted {err*1e3:.4f} mm (stale head frame)"

    def worst_penetration() -> float:
        worst = 0.0
        for c in range(data.ncon):
            con = data.contact[c]
            if toy_g in (con.geom1, con.geom2):
                worst = min(worst, float(con.dist))
        return worst

    reset_home()
    x_head, R_head = head_frame()
    mouth_in_head = R_head.T @ (data.site_xpos[mouth] - x_head)
    # Head-frame axes, as world directions (columns of R).
    down_h = -R_head.T @ np.array([0.0, 0.0, 1.0])

    print("=== HEAD FRAME ===")
    print(f"  world UP      in head coords: {np.round(R_head.T @ [0, 0, 1.0], 3)}")
    print(f"  world FORWARD in head coords: {np.round(R_head.T @ [1.0, 0, 0], 3)}")
    print(f"  mouth_tip in head frame     : {np.round(mouth_in_head * 1e3, 2)} mm")
    print(f"  toy half-extents            : {np.round(model.geom_size[toy_g] * 1e3, 1)} mm")

    print("\n=== PENETRATION SWEEP (toy centre below mouth_tip) ===")
    print(f"  {'down (mm)':>10} {'worst contact dist (mm)':>26}")
    first_clear = None
    for down_mm in np.arange(0.0, 32.0, 1.0):
        reset_home()
        place(mouth_in_head + down_h * (down_mm / 1e3))
        worst = worst_penetration()
        if worst >= -1e-6 and first_clear is None:
            first_clear = down_mm
        mark = "  <- first clear" if first_clear == down_mm else ""
        if down_mm % 2 == 0 or mark:
            print(f"  {down_mm:10.0f} {worst * 1e3:26.2f}{mark}")

    offset = mouth_in_head + down_h * (first_clear / 1e3)
    print(f"\n  CARRY_OFFSET_IN_HEAD = ({offset[0]:.6f}, {offset[1]:.6f}, {offset[2]:.6f})")

    print("\n=== CoM COUPLING at that carry offset ===")
    print(f"  {'toy mass':>9} {'CoM x (mm)':>12} {'shift (mm)':>12} {'% of sole':>11}")
    coms = {}
    for mass in (1e-9, 0.010, 0.020, 0.030, 0.050, 0.080):
        reset_home()
        model.body_mass[toy_b] = mass
        place(offset)
        m = model.body_mass[1:]
        coms[mass] = float((m[:, None] * data.xipos[1:]).sum(0)[0] / m.sum())
    base = coms[1e-9]
    for mass, cx in coms.items():
        if mass < 1e-6:
            print(f"  {'none':>9} {cx * 1e3:12.3f} {'--':>12} {'--':>11}")
        else:
            shift = cx - base
            print(
                f"  {mass * 1e3:8.0f}g {cx * 1e3:12.3f} {shift * 1e3:12.3f} "
                f"{shift / SOLE_LENGTH_M * 100:10.1f}%"
            )
    print(
        "\n  For reference, Phase 1 (scripts/payload_sweep.py) measured 8.2 mm at 80 g\n"
        "  for the grasp pose; the carry pose hangs closer to the body."
    )


if __name__ == "__main__":
    main()
