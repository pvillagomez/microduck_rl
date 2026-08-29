"""Cfg-invariant and mechanism tests for the GraspLift task (roadmap Phase 1).

All CPU, no GPU: the config assertions lock in the reward-sign and phase-gating
invariants, and the mechanism tests compile the real scene to check the grasp weld
is installed correctly and that the weld pose maths actually pins the toy to the
beak. The physics of the weld itself is checked in
``test_grasp_weld_holds_object`` by stepping plain MuJoCo.
"""

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjlab_microduck.tasks.microduck_grasp_lift_env_cfg import (
    DESCENT_END,
    GRASP_MAX_REL_SPEED,
    GRASP_MIN_ALIGNMENT,
    GRASP_RADIUS,
    GL_PERIOD,
    HOLD_END,
    RISE_END,
    TOY_CARRY_STD,
    TOY_CARRY_Z,
    TOY_HALF_HEIGHT,
    TOY_MASS_RANGE_KG,
    TOY_NOMINAL_KG,
    make_microduck_grasp_lift_env_cfg,
)
from mjlab_microduck.tasks.mdp import GRASP_WELD_NAME, GroundPickPhaseCommand


# ── Config invariants ────────────────────────────────────────────────────────


def test_grasp_lift_cfg_builds_and_registers_toy():
    cfg = make_microduck_grasp_lift_env_cfg()
    # The robot MUST stay first: set_random_ground_state and the base reset events
    # write robot root state at qpos[:, 0:7].
    assert list(cfg.scene.entities.keys()) == ["robot", "toy"]
    assert cfg.scene.spec_fn is not None, "grasp weld spec_fn must be installed"


def test_grasp_lift_play_variant_builds():
    cfg = make_microduck_grasp_lift_env_cfg(play=True)
    assert "grasp_engage" in cfg.rewards


def test_grasp_lift_command_is_phase():
    cfg = make_microduck_grasp_lift_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.class_type is GroundPickPhaseCommand
    assert cmd.period == GL_PERIOD


def test_phase_profile_is_ordered():
    """Segments must be strictly increasing, else the gates silently invert."""
    assert 0.0 < DESCENT_END < HOLD_END < RISE_END < 1.0


def test_grasp_lift_task_rewards_wired():
    cfg = make_microduck_grasp_lift_env_cfg()
    r = cfg.rewards

    # Reach the toy (not the floor: ground_pick's target was the ground).
    assert r["mouth_toy_proximity"].weight == 3.0
    assert r["mouth_toy_proximity"].params["asset_name"] == "toy"
    assert r["mouth_perpendicular"].weight == 2.0

    # The grasp itself.
    assert r["grasp_engage"].weight > 0.0
    assert r["grasp_held"].weight > 0.0
    assert r["toy_lift"].weight > 0.0
    assert r["toy_carried"].params["target_height"] == TOY_CARRY_Z

    # Return-to-stand stack, gated on the rise.
    for name in ("return_pose_legs", "return_pose_neck", "return_upright"):
        assert r[name].params["rise_end"] == RISE_END

    # ground_pick's simulated payload must NOT be here: the point of this task is
    # that the payload is a real welded body, not an external wrench.
    assert "mouth_payload_force" not in r
    assert "sample_mouth_payload" not in cfg.events


def test_penalty_terms_have_costing_signs():
    """Every penalty must actually cost.

    mdp.py has two penalty styles: mjlab-base cost functions return >= 0 and take a
    NEGATIVE weight, while self-negating microduck ``*_penalty`` helpers return <= 0
    and take a POSITIVE weight. A negative weight on a self-negating penalty
    double-negates into a reward for the violation, which the policy will farm.
    """
    cfg = make_microduck_grasp_lift_env_cfg()
    for name in (
        "toy_knocked_away",   # returns >= 0
        "neck_vel_descent",   # returns >= 0
        "feet_flat",          # returns >= 0
        "head_impact_penalty",
        "self_collisions",
        "action_rate_l2",
        "neck_action_rate_l2",
        "joint_torques_l2",
        "body_ang_vel",
        "angular_momentum",
    ):
        assert cfg.rewards[name].weight < 0.0, f"{name} must have a negative weight"


def test_lift_reward_is_potential_based_and_capped():
    """The lift term pays a per-step DELTA, clipped, not an altitude level.

    A height-LEVEL reward pays every step at altitude, so arriving early and hard is
    the argmax; a clipped signed delta makes holding pay zero and lowering refund.
    """
    cfg = make_microduck_grasp_lift_env_cfg()
    lift = cfg.rewards["toy_lift"]
    assert lift.params["max_step"] > 0.0
    # Per-step payout stays bounded and comparable to the standing stack.
    assert lift.weight * lift.params["max_step"] <= 5.0


def test_toy_carry_target_is_reachable_and_loose():
    """The carry target must sit within the beak's standing height, with a std wide
    enough to cover the unescapable latch offset (the toy is welded wherever it was
    caught, anywhere within GRASP_RADIUS of the mouth)."""
    assert TOY_CARRY_STD >= GRASP_RADIUS
    assert 0.15 < TOY_CARRY_Z < 0.25


def test_actor_stays_blind_to_the_toy():
    """The real robot has no toy sensing until roadmap Phase 4 (perception)."""
    cfg = make_microduck_grasp_lift_env_cfg()
    actor = cfg.observations["actor"].terms
    critic = cfg.observations["critic"].terms
    assert not any("toy" in name or "grasp" in name for name in actor)
    for name in ("toy_position", "toy_velocity", "grasp_state"):
        assert name in critic
    # Unified 13D command block: twist(3) + head(4) + body(6), head/body zero-padded.
    assert actor["head_command"].params["dim"] == 4
    assert actor["body_command"].params["dim"] == 6


def test_grasp_events_are_wired_and_ordered():
    cfg = make_microduck_grasp_lift_env_cfg()
    events = cfg.events

    assert events["grasp_latch"].mode == "step"
    assert events["reset_grasp"].mode == "reset"
    assert events["reset_toy"].mode == "reset"

    # The toy spawn is derived from the final robot pose, and event terms fire in
    # dict insertion order, so reset_toy must come after reset_base.
    order = list(events.keys())
    assert order.index("reset_toy") > order.index("reset_base")

    # reset_grasp carries the per-world eq_data expansion for the whole task.
    # Without it every env silently shares env 0's weld pose.
    assert "eq_data" in getattr(events["reset_grasp"].func, "model_fields", ())


def test_grasp_gate_params_are_physical():
    cfg = make_microduck_grasp_lift_env_cfg()
    p = cfg.events["grasp_latch"].params
    assert p["radius"] == GRASP_RADIUS
    assert p["min_alignment"] == GRASP_MIN_ALIGNMENT
    assert p["max_rel_speed"] == GRASP_MAX_REL_SPEED
    assert p["sensor_name"] == "head_toy_contact"
    # A contact sensor must exist for that name, or the gate degrades to proximity
    # only and the policy can "grasp" without ever touching the toy.
    assert any(s.name == "head_toy_contact" for s in cfg.scene.sensors)


def test_toy_mass_randomization_spans_the_measured_range():
    """dr.pseudo_inertia takes alpha = ln(scale)/2 and scales mass AND inertia."""
    cfg = make_microduck_grasp_lift_env_cfg()
    lo_alpha, hi_alpha = cfg.events["randomize_toy_mass"].params["alpha_range"]
    lo_kg = TOY_NOMINAL_KG * math.exp(2 * lo_alpha)
    hi_kg = TOY_NOMINAL_KG * math.exp(2 * hi_alpha)
    assert lo_kg == pytest.approx(TOY_MASS_RANGE_KG[0], rel=1e-6)
    assert hi_kg == pytest.approx(TOY_MASS_RANGE_KG[1], rel=1e-6)


def test_flat_terrain_only():
    """A toy on generated rough terrain has no known spawn height."""
    cfg = make_microduck_grasp_lift_env_cfg()
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.scene.terrain.terrain_generator is None
    assert "terrain_levels" not in cfg.curriculum


# ── Mechanism tests (compile the real scene) ─────────────────────────────────


def _compiled_scene():
    from copy import deepcopy

    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains.terrain_entity import TerrainEntityCfg

    from mjlab_microduck.robot.microduck_constants import (
        MICRODUCK_GRASP_LIFT_ROBOT_CFG,
        MICRODUCK_TOY_CFG,
    )
    from mjlab_microduck.tasks.mdp import make_grasp_weld_spec_fn

    # Deepcopy the entity cfgs: building an Entity edits its cfg in place (spec
    # editors, actuators, collisions), and these are the module-level singletons
    # the registered tasks also reference. Without this, compiling a scene here
    # leaks into every later test — it silently broke head/toy contact and made
    # the integration test below fail only when run after these.
    entities = {
        "robot": deepcopy(MICRODUCK_GRASP_LIFT_ROBOT_CFG),
        "toy": deepcopy(MICRODUCK_TOY_CFG),
    }
    scene = Scene(
        SceneCfg(
            num_envs=1,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities=entities,
            spec_fn=make_grasp_weld_spec_fn(),
        ),
        device="cpu",
    )
    return scene.compile()


def test_grasp_weld_is_installed_inactive():
    model = _compiled_scene()
    eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, GRASP_WELD_NAME)
    assert eq >= 0, "grasp weld missing from the compiled scene"
    assert model.eq_type[eq] == mujoco.mjtEq.mjEQ_WELD
    # Must start OFF: the toy is not glued to the beak at spawn.
    assert not model.eq_active0[eq]
    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/jaw_soft")
    toy = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "toy/toy")
    assert model.eq_obj1id[eq] == head
    assert model.eq_obj2id[eq] == toy


def test_toy_matches_the_cfg_constants():
    """toy.xml and the env cfg must agree, or the mass DR and spawn height drift."""
    model = _compiled_scene()
    toy = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "toy/toy")
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "toy/toy_geom")
    assert model.body_mass[toy] == pytest.approx(TOY_NOMINAL_KG)
    assert model.geom_size[geom][2] == pytest.approx(TOY_HALF_HEIGHT)


def test_toy_fits_the_measured_mouth_reach():
    """The toy's resting centre must be inside the beak's vertical reach.

    Measured with both feet grounded, mouth_tip bottoms out at z ~= 13mm. A block
    whose centre sits above that can never be reached, and no reward tuning fixes
    it — this guards a re-export of the robot model silently breaking the task.
    """
    model = _compiled_scene()
    data = mujoco.MjData(model)
    qadr = lambda j: model.jnt_qposadr[  # noqa: E731
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    ]
    data.qpos[:] = model.qpos0
    for side, sign in (("left", 1), ("right", -1)):
        data.qpos[qadr(f"robot/{side}_hip_pitch")] = sign * -0.5
        data.qpos[qadr(f"robot/{side}_knee")] = sign * -2.0
        data.qpos[qadr(f"robot/{side}_ankle")] = sign * 0.591
    data.qpos[qadr("robot/neck_pitch")] = -1.57
    data.qpos[qadr("robot/head_pitch")] = 0.523
    mujoco.mj_forward(model, data)

    feet = [
        data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"robot/{s}_foot")][2]
        for s in ("left", "right")
    ]
    mouth_above_floor = (
        data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "robot/mouth_tip")][2]
        - min(feet)
    )
    assert mouth_above_floor <= TOY_HALF_HEIGHT + GRASP_RADIUS


def test_grasp_weld_holds_object():
    """The weld pose maths must actually pin the toy to the beak.

    This is the core mechanism of the task: ``update_grasp_latch`` writes
    ``eq_data`` as [anchor=0 | R_head^T (x_toy - x_head) | q_head^-1 q_toy | 1], and
    if any of those terms is wrong the toy either teleports on latch or drifts out
    of the beak. Reproduced here in plain MuJoCo and stepped.
    """
    model = _compiled_scene()
    data = mujoco.MjData(model)
    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/jaw_soft")
    toy = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "toy/toy")
    eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, GRASP_WELD_NAME)
    mouth = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "robot/mouth_tip")
    toy_adr = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "toy/toy_free")
    ]

    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    data.qpos[toy_adr : toy_adr + 3] = data.site_xpos[mouth]
    mujoco.mj_forward(model, data)

    rot = data.xmat[head].reshape(3, 3)
    head_inv, relpose = np.empty(4), np.empty(4)
    mujoco.mju_negQuat(head_inv, data.xquat[head])
    mujoco.mju_mulQuat(relpose, head_inv, data.xquat[toy])
    rel0 = rot.T @ (data.xpos[toy] - data.xpos[head])

    model.eq_data[eq, 0:3] = 0.0
    model.eq_data[eq, 3:6] = rel0
    model.eq_data[eq, 6:10] = relpose
    model.eq_data[eq, 10] = 1.0
    data.eq_active[eq] = 1

    for _ in range(1000):
        mujoco.mj_step(model, data)

    drift = np.linalg.norm(
        data.xmat[head].reshape(3, 3).T @ (data.xpos[toy] - data.xpos[head]) - rel0
    )
    assert np.all(np.isfinite(data.qpos))
    # Not exactly rigid on purpose: the weld is deliberately compliant (soft beak).
    assert drift < 0.005, f"toy slipped {drift * 1000:.1f} mm out of the beak"

    # And releasing it must actually drop the toy.
    data.eq_active[eq] = 0
    for _ in range(1500):
        mujoco.mj_step(model, data)
    assert data.xpos[toy][2] < 0.03, "toy did not fall after the weld was released"


# ── Backlash variant ─────────────────────────────────────────────────────────


def test_backlash_variant_keeps_toy_and_weld():
    """The backlash twin must keep the toy entity AND a resolvable grasp weld.

    ``make_backlash_variant`` swaps only the ``robot`` key, so a second entity and
    ``scene.spec_fn`` survive by construction — but nothing else in the repo has a
    multi-entity backlash task, so this locks that in. Compiling the scene is the
    part that matters: the weld addresses bodies by prefixed name, so if the
    backlash export ever renamed ``jaw_soft`` the equality would silently fail to
    resolve and the task would train with no grasp at all.
    """
    from mjlab.scene import Scene, SceneCfg
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.terrains.terrain_entity import TerrainEntityCfg

    cfg = load_env_cfg("Mjlab-GraspLift-Flat-Backlash-MicroDuck")
    # Robot MUST stay first: base reset events write robot root at qpos[:, 0:7].
    assert list(cfg.scene.entities.keys()) == ["robot", "toy"]
    assert cfg.scene.spec_fn is not None

    # cfg came from load_env_cfg, which deep-copies, so building a Scene from it
    # cannot leak into the shared entity cfgs (see _compiled_scene).
    model = Scene(
        SceneCfg(
            num_envs=1,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities=cfg.scene.entities,
            spec_fn=cfg.scene.spec_fn,
        ),
        device="cpu",
    ).compile()

    eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, GRASP_WELD_NAME)
    assert eq >= 0, "grasp weld did not resolve on the backlash model"
    assert model.eq_obj1id[eq] == mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "robot/jaw_soft"
    )
    assert model.eq_obj2id[eq] == mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "toy/toy"
    )
    assert not model.eq_active0[eq]

    # Backlash joints ride their hard limits by design and must be excluded from
    # the soft-limit penalty, or it fires permanently.
    assert cfg.rewards["dof_pos_limits"].params["asset_cfg"].joint_names == (
        r"^(?!passive_).*",
    )


def test_backlash_variant_mirrors_base_robot_model():
    """AGENTS.md invariant: a -Backlash- variant mirrors its base task's model.

    GraspLift runs on allcollisions, so its twin must use the allcollisions
    backlash robot — a walk-model twin would confound any backlash A/B with a
    collision-model change.
    """
    from mjlab.tasks.registry import load_env_cfg

    from mjlab_microduck.robot.microduck_constants import get_backlash_spec

    cfg = load_env_cfg("Mjlab-GraspLift-Flat-Backlash-MicroDuck")
    # load_env_cfg deep-copies, so identity against the cfg object is meaningless;
    # spec_fn survives the copy and pins the actual XML that gets compiled.
    assert cfg.scene.entities["robot"].spec_fn is get_backlash_spec


def test_backlash_variant_preserves_the_61d_obs_contract():
    """Obs/action dims must be unchanged: 14 servo joints, 61 D actor.

    The backlash model carries 28 joints (14 servo + 14 passive backlash hinges);
    if those leaked into the joint obs the actor would blow past 61 D and stop
    being hot-swappable against the rest of the policy family.
    """
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-GraspLift-Flat-Backlash-MicroDuck")
    for group in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            sel = cfg.observations[group].terms[term].params["asset_cfg"].joint_names
            assert sel == (r"^(?!passive_).*",), (
                f"{group}.{term} would include the backlash hinges"
            )


def test_grasp_weld_data_satisfies_the_weld_constraint():
    """The SHIPPED weld maths must zero MuJoCo's weld residual.

    ``update_grasp_latch`` writes ``eq_data`` via ``_grasp_weld_data``; this checks
    that exact function against MuJoCo's own definition of a satisfied weld:

        pos1 = x_head + R_head @ data[3:6]  ==  pos2 = x_toy + R_toy @ data[0:3]
        q_head * data[6:10]                 ==  q_toy

    Deliberately asserts the CONSTRAINT rather than re-deriving the terms. An
    earlier version of this file recomputed the anchors independently, which meant
    a wrong call site in the shipped code still passed — mutation testing caught
    that (breaking the anchor write went undetected).
    """
    import torch
    from mjlab.utils.lab_api.math import matrix_from_quat, quat_mul

    from mjlab_microduck.tasks.mdp import _grasp_weld_data

    torch.manual_seed(0)
    n = 16
    q_head = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
    q_toy = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
    x_head = torch.randn(n, 3)
    x_toy = torch.randn(n, 3)

    data = _grasp_weld_data(q_head, x_head, q_toy, x_toy)
    assert data.shape == (n, 11)
    assert torch.allclose(data[:, 10], torch.ones(n)), "torquescale must be 1 (6-DoF weld)"

    r_head = matrix_from_quat(q_head)
    r_toy = matrix_from_quat(q_toy)
    pos1 = x_head + torch.bmm(r_head, data[:, 3:6].unsqueeze(-1)).squeeze(-1)
    pos2 = x_toy + torch.bmm(r_toy, data[:, 0:3].unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(pos1, pos2, atol=1e-5), "weld position residual is non-zero"

    q_pred = quat_mul(q_head, data[:, 6:10])
    # Quaternions double-cover rotations, so q and -q are the same orientation.
    aligned = torch.minimum((q_pred - q_toy).abs().max(dim=-1).values,
                            (q_pred + q_toy).abs().max(dim=-1).values)
    assert torch.all(aligned < 1e-5), "weld orientation residual is non-zero"


def test_latch_engages_and_the_toy_physically_follows_the_beak():
    """End-to-end: stepping the env must actually engage the weld and hold the toy.

    Slower than the rest of this file (it instantiates the env) but it covers the
    one thing config assertions cannot: the runtime side-effect of the latch. If
    ``update_grasp_latch`` wrote ``eq_data`` and never set ``eq_active``, every
    config test here would still pass while the grasp silently never engaged — a
    task that can never succeed, discovered only after paying for a training run.
    Mutation testing found exactly that hole.

    Drives the real integration path (``env.step`` -> registered "step" event)
    rather than calling the latch directly, because the contact gate reads sensor
    data that only refreshes inside the decimation loop.
    """
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from mjlab_microduck.tasks.mdp import _grasp_eq_id, _grasp_flag

    cfg = load_env_cfg("Mjlab-GraspLift-Flat-MicroDuck")
    cfg.scene.num_envs = 2
    # Seed it: spawn placement, CoM randomization and pushes are all random, and
    # an unseeded run makes this test flaky rather than wrong.
    cfg.seed = 0
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        env.reset()
        action = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
        for _ in range(3):
            env.step(action)

        robot, toy = env.scene["robot"], env.scene["toy"]
        joints, sites = list(robot.joint_names), list(robot.site_names)
        qadr = robot.indexing.joint_q_adr
        mouth = sites.index("mouth_tip")

        # Point the beak at the floor and place the toy in it: satisfies the
        # proximity, alignment and (via the resulting overlap) contact gates.
        for name, value in (("neck_pitch", -1.35), ("head_pitch", 0.30)):
            env.sim.data.qpos[:, qadr[joints.index(name)]] = value
        env.sim.forward()
        pose = torch.zeros(env.num_envs, 7)
        pose[:, :3] = robot.data.site_pos_w[:, mouth, :]
        pose[:, 3] = 1.0
        toy.write_root_link_pose_to_sim(pose)
        toy.write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6))
        env.sim.forward()

        assert not bool(_grasp_flag(env).any()), "nothing should be held yet"

        # Hold the toy in the beak for a few steps rather than one, and give it the
        # BEAK's velocity rather than zero. Under zero actions the robot is falling
        # at ~0.5 m/s, and a toy pinned at zero velocity is something the head is
        # slapping through — which the latch's relative-speed gate correctly
        # refuses. Matching velocities is the physically meaningful setup: beak and
        # object moving together, in contact, mouth down.
        for _ in range(10):
            pose[:, :3] = robot.data.site_pos_w[:, mouth, :]
            vel = torch.zeros(env.num_envs, 6)
            vel[:, :3] = robot.data.site_lin_vel_w[:, mouth, :]
            toy.write_root_link_pose_to_sim(pose)
            toy.write_root_link_velocity_to_sim(vel)
            env.sim.forward()
            env.step(action)
            if bool(_grasp_flag(env).all()):
                break

        assert bool(_grasp_flag(env).all()), "the latch did not engage"
        assert bool(
            env.sim.data.eq_active[:, _grasp_eq_id(env)].all()
        ), "eq_active was never set — the weld is inert"

        # The decisive physical check: with the weld truly active the toy tracks
        # the beak. Guarded on episode end — the robot is falling under zero
        # actions, and a fell_over reset legitimately RELEASES the weld and
        # re-spawns the toy on the floor, which would read as a huge false "slip".
        offset = (toy.data.root_link_pos_w - robot.data.site_pos_w[:, mouth, :]).clone()
        steps = 0
        for _ in range(8):
            _, _, terminated, timed_out, _ = env.step(action)
            if bool((terminated | timed_out).any()):
                break
            steps += 1
            drift = torch.linalg.norm(
                (toy.data.root_link_pos_w - robot.data.site_pos_w[:, mouth, :]) - offset,
                dim=-1,
            )
            assert bool((drift < 0.01).all()), f"toy slipped out of the beak: {drift}"
        assert steps > 0, "episode ended before the weld could be observed holding"
        assert bool(torch.isfinite(env.sim.data.qpos).all())
    finally:
        env.close()
