"""Measure how much mass the Microduck can actually hold in its beak.

Roadmap Phase 1 flags this as a blocker: ``ground_pick`` fakes a 10-40 g payload
with an external wrench, but real dog toys are 50-200 g+ on an ~740 g robot, and
nobody had measured where the neck servos give out. This script answers that on
CPU MuJoCo (no GPU, no training), so the training mass range is a measurement
rather than a guess.

What it measures
----------------
Static torque feasibility. For a pose held still, the equation of motion reduces
to ``tau + J^T F_payload = qfrc_bias``, so the torque each servo must produce is::

    tau_required = qfrc_bias - J_mouth^T @ (0, 0, -m g)

``qfrc_bias`` already carries the robot's own gravity; the payload enters through
the mouth-tip site Jacobian as a point mass hanging off the beak. Each servo's
requirement is then compared against the model's real ``actuator_forcerange``,
which mjlab sets from the BAM XL330 model (+/-1.07 Nm) rather than the placeholder
MJCF position gains.

This is deliberately NOT a dynamic drop test. The compiled CPU model has BAM's
torque actuators but not BAM's Python-side control law, so free-running it would
measure a controller that does not exist. Static feasibility needs no controller
and gives a hard physical bound: a mass whose required torque exceeds stall cannot
be held by any policy, however well trained.

Poses are evaluated with the feet resting on the floor:
  * ``stand``  - HOME pose, head up: the carry pose at the end of the lift.
  * ``crouch`` - mouth near the floor: the grasp instant, worst neck lever arm.

Caveat on ``crouch``: it comes from a purely kinematic reach scan, so the ankles
are tilted and only a ~5mm strip of each sole touches down. Its torque numbers are
sound (they do not depend on the contact patch), but its ``% sole`` column is
against that tilted strip, not against a foot the robot is standing flat on. Read
the ``stand`` row for the balance picture.

Usage:
    uv run python scripts/payload_sweep.py
    uv run python scripts/payload_sweep.py --masses-g 10 30 60 100 --margin 0.7
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

from mjlab.scene import Scene, SceneCfg
from mjlab.terrains.terrain_entity import TerrainEntityCfg

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_GRASP_LIFT_ROBOT_CFG,
    MICRODUCK_TOY_CFG,
)
from mjlab_microduck.tasks.mdp import make_grasp_weld_spec_fn

GRAVITY = 9.81

# Deep crouch that puts mouth_tip on the floor, taken from the reach scan quoted
# in microduck_grasp_lift_env_cfg.py's header. Leg joints are mirrored left/right.
CROUCH_POSE = {
    "hip_pitch": -0.5,
    "knee": -2.0,
    "ankle": 0.591,
    "neck_pitch": -1.57,
    "head_pitch": 0.523,
}


def build_model() -> mujoco.MjModel:
    """Compile the GraspLift scene (robot + toy + inactive grasp weld) for CPU MuJoCo."""
    scene = Scene(
        SceneCfg(
            num_envs=1,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": MICRODUCK_GRASP_LIFT_ROBOT_CFG, "toy": MICRODUCK_TOY_CFG},
            spec_fn=make_grasp_weld_spec_fn(),
        ),
        device="cpu",
    )
    return scene.compile()


def _qadr(model: mujoco.MjModel, joint: str) -> int:
    return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]


def set_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: str) -> None:
    """Write a named pose into qpos and drop it so the lowest foot rests on the floor."""
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    if pose == "crouch":
        for side, sign in (("left", 1), ("right", -1)):
            data.qpos[_qadr(model, f"robot/{side}_hip_pitch")] = sign * CROUCH_POSE["hip_pitch"]
            data.qpos[_qadr(model, f"robot/{side}_knee")] = sign * CROUCH_POSE["knee"]
            data.qpos[_qadr(model, f"robot/{side}_ankle")] = sign * CROUCH_POSE["ankle"]
        data.qpos[_qadr(model, "robot/neck_pitch")] = CROUCH_POSE["neck_pitch"]
        data.qpos[_qadr(model, "robot/head_pitch")] = CROUCH_POSE["head_pitch"]
    mujoco.mj_forward(model, data)



def required_torques(model: mujoco.MjModel, data: mujoco.MjData, mass_kg: float) -> np.ndarray:
    """Static servo torques needed to hold the current pose with ``mass_kg`` on the beak."""
    mouth = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "robot/mouth_tip")
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, mouth)

    force = np.array([0.0, 0.0, -mass_kg * GRAVITY])
    tau_full = data.qfrc_bias - jacp.T @ force

    # Map each actuator to its dof. The freejoint dofs are reacted by the ground,
    # not by a servo, so only actuated dofs are checked.
    out = np.zeros(model.nu)
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        out[i] = tau_full[model.jnt_dofadr[jid]]
    return out


def _foot_geom_ids(model: mujoco.MjModel) -> list[int]:
    ids = []
    for name in ("robot/left_foot_collision", "robot/right_foot_collision"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise RuntimeError(f"Foot collision geom '{name}' not found.")
        ids.append(gid)
    return ids


def _foot_vertices_w(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """World-frame vertices of both foot collision meshes."""
    out = []
    for gid in _foot_geom_ids(model):
        mesh = model.geom_dataid[gid]
        adr, num = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        verts = model.mesh_vert[adr : adr + num].astype(np.float64)
        rot = data.geom_xmat[gid].reshape(3, 3)
        out.append(verts @ rot.T + data.geom_xpos[gid])
    return np.concatenate(out, axis=0)


def seat_on_floor(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Drop the trunk so the lowest foot vertex rests at z=0; return the support patch.

    The support polygon is taken from the foot MESH footprint, not from
    ``data.contact``: MuJoCo's mesh-plane collision emits a single contact point per
    foot, so a contact-derived polygon degenerates to a line and reports every pose
    (even an unloaded stand) as tipping.

    Returns the XY of every foot vertex within ``patch`` of the floor — the part of
    the sole that actually bears load.
    """
    verts = _foot_vertices_w(model, data)
    data.qpos[2] -= verts[:, 2].min()
    mujoco.mj_forward(model, data)

    verts = _foot_vertices_w(model, data)
    patch = 0.002
    return verts[verts[:, 2] <= verts[:, 2].min() + patch][:, :2]


def com_shift(
    model: mujoco.MjModel, data: mujoco.MjData, mass_kg: float
) -> float:
    """Horizontal distance (m) the payload moves the whole-body CoM.

    Reported as a DIAGNOSTIC rather than a pass/fail. A static "is the CoM inside
    the support polygon" test is the wrong criterion for this robot: measured at
    HOME, the unloaded Microduck's CoM already sits ~3mm ahead of the front edge of
    its soles (the STAND2 keyframe deliberately puts the CoM over the ankle axis,
    and the sole extends behind it). It stands by actively balancing, so a static
    test flags even the unloaded robot as tipping.

    What the payload mass does control is how much EXTRA disturbance the balance
    controller has to absorb, and that is exactly this number: the toy is treated as
    a point mass at the mouth tip, and this is how far it drags the CoM forward.
    """
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk_base")
    mouth = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "robot/mouth_tip")
    m_robot = float(model.body_subtreemass[root])

    unloaded = data.subtree_com[root][:2]
    loaded = (m_robot * unloaded + mass_kg * data.site_xpos[mouth][:2]) / (
        m_robot + mass_kg
    )
    return float(np.linalg.norm(loaded - unloaded))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--masses-g",
        type=float,
        nargs="+",
        default=[10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200],
        help="Payload masses to test, in grams.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.7,
        help="Usable fraction of stall torque. Servos cannot be held at 100%% stall, "
        "and the policy needs headroom to MOVE the load, not merely hold it.",
    )
    parser.add_argument("--poses", nargs="+", default=["stand", "crouch"])
    args = parser.parse_args()

    model = build_model()
    data = mujoco.MjData(model)
    limit = float(model.actuator_forcerange[0][1])
    usable = limit * args.margin
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i).replace("robot/", "")
        for i in range(model.nu)
    ]
    robot_mass = sum(
        model.body_mass[i]
        for i in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("robot/")
    )

    print(f"Robot mass           : {robot_mass * 1000:.0f} g")
    print(f"Servo stall torque   : {limit:.3f} Nm (BAM XL330, from actuator_forcerange)")
    print(f"Usable at {args.margin:.0%} margin  : {usable:.3f} Nm")

    verdicts: dict[str, float | None] = {}
    for pose in args.poses:
        set_pose(model, data, pose)
        support = seat_on_floor(model, data)
        sole_len = float(support[:, 0].max() - support[:, 0].min())
        mouth_z = data.site_xpos[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "robot/mouth_tip")
        ][2]
        trunk_z = data.xpos[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk_base")
        ][2]
        print(
            f"\n=== pose '{pose}' : trunk z = {trunk_z * 1000:.1f} mm, "
            f"mouth_tip z = {mouth_z * 1000:.1f} mm, "
            f"sole contact length = {sole_len * 1000:.1f} mm ==="
        )
        print(
            f"{'mass (g)':>9} {'worst servo':>16} {'torque (Nm)':>12} "
            f"{'% stall':>8} {'CoM shift':>10} {'% sole':>7} {'verdict':>9}"
        )

        heaviest = None
        for grams in args.masses_g:
            kg = grams / 1000.0
            tau = required_torques(model, data, kg)
            worst = int(np.argmax(np.abs(tau)))
            peak = float(abs(tau[worst]))
            shift = com_shift(model, data, kg)
            ok = peak <= usable
            if ok:
                heaviest = grams
            print(
                f"{grams:9.0f} {names[worst]:>16} {peak:12.3f} "
                f"{100 * peak / limit:7.0f}% {shift * 1000:9.1f}mm "
                f"{100 * shift / sole_len:6.0f}% {'ok' if ok else 'OVER':>9}"
            )
        verdicts[pose] = heaviest

    print("\n--- summary ---")
    for pose, heaviest in verdicts.items():
        if heaviest is None:
            print(f"  {pose:7s}: torque limit exceeded at every tested mass")
        else:
            print(f"  {pose:7s}: torque-feasible up to {heaviest:.0f} g")
    binding = [v for v in verdicts.values() if v is not None]
    if binding and len(binding) == len(verdicts):
        print(
            f"\nServo torque is not the binding constraint below {min(binding):.0f} g. "
            "What actually limits the payload is BALANCE: see the CoM-shift column, "
            "which is the extra disturbance the policy has to absorb, as a fraction "
            "of the sole it has to keep its weight over. Torque feasibility is also a "
            "STATIC bound - accelerating the toy during the lift costs more on top."
        )


if __name__ == "__main__":
    main()
