"""Cfg-invariant and mechanism tests for the Carry task (roadmap Phase 2).

All CPU, no GPU. The config assertions lock in the invariants that make this
Phase 2 rather than a second copy of Phase 1 — real velocity commands in the
twist slot, the gait reward stack, no latch — and the mechanism tests instantiate
the real env to check the thing config assertions cannot: that the toy actually
starts welded at the beak and stays there while the robot moves.

Every assertion here was mutation-checked: each invariant was deliberately broken
and the corresponding test confirmed to fail. See the PR description for the table.
"""

import math

import numpy as np
import pytest
import torch

from mjlab_microduck.tasks.microduck_carry_env_cfg import (
    CARRY_OFFSET_IN_HEAD,
    PAYLOAD_FREE_ACCEL,
    PAYLOAD_VIOLENCE_WEIGHT,
    TOY_MASS_RANGE_KG,
    TOY_NOMINAL_KG,
    make_microduck_carry_env_cfg,
)


# ── Config invariants ────────────────────────────────────────────────────────


def test_carry_cfg_builds_and_registers_toy():
    cfg = make_microduck_carry_env_cfg()
    # The robot MUST stay first: the base reset events write robot root state at
    # qpos[:, 0:7].
    assert list(cfg.scene.entities.keys()) == ["robot", "toy"]
    assert cfg.scene.spec_fn is not None, "grasp weld spec_fn must be installed"


def test_carry_play_variant_builds():
    cfg = make_microduck_carry_env_cfg(play=True)
    assert "attach_toy" in cfg.events


def test_twist_slot_carries_real_velocity_commands_not_a_phase():
    """THE Phase 2 constraint.

    Phase 1 spends the twist command slot on its pick phase
    (``GroundPickPhaseCommand`` emitting ``[cos, sin, 0]``). Phase 2 needs that
    slot for actual velocity commands, and the shared 61 D obs contract forbids
    adding one. If this ever regresses to a phase command, the task silently stops
    being a locomotion task while every other test still passes.
    """
    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

    from mjlab_microduck.tasks.mdp import GroundPickPhaseCommandCfg

    cfg = make_microduck_carry_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, UniformVelocityCommandCfg)
    # NOT merely "is a UniformVelocityCommandCfg": GroundPickPhaseCommandCfg
    # SUBCLASSES that, so the isinstance check above passes for the phase command
    # too and is nearly vacuous on its own. This is the assertion that bites.
    assert not isinstance(cmd, GroundPickPhaseCommandCfg), (
        "the twist slot has regressed to Phase 1's pick-phase encoding; Phase 2 "
        "needs it for real velocity commands and the 61 D obs contract has no "
        "spare slot"
    )
    # And it must actually command motion, or "carry while walking" is just
    # "stand while holding".
    assert cmd.ranges.lin_vel_x[1] > 0.0
    assert cmd.ranges.ang_vel_z[1] > 0.0


def test_carry_inherits_the_gait_reward_stack():
    """Proves this is built on the velocity recipe, not the phased pick stack."""
    cfg = make_microduck_carry_env_cfg()
    for name in ("track_linear_velocity", "track_angular_velocity", "air_time"):
        assert name in cfg.rewards, f"missing gait reward {name}"
        assert cfg.rewards[name].weight > 0.0

    # ground_pick / grasp_lift phase machinery must NOT be here.
    for name in ("mouth_toy_proximity", "grasp_engage", "grasp_held", "toy_lift"):
        assert name not in cfg.rewards, f"Phase 1 term {name} leaked into Phase 2"


def test_head_pose_bias_ema_term_is_inherited():
    """The AGENTS.md payload lesson: price the escapable DC bias, not oscillation.

    A tight INSTANTANEOUS head-tracking std once taxed walking so hard the policy
    stood still. The velocity recipe's answer is an L1 penalty on a 1 s EMA of the
    head error, and a payload makes that term MORE relevant, not less — carrying
    mass on the beak is exactly what produces a steady droop. Losing it (or having
    it degenerate to an instantaneous term) is the known way to break this task.
    """
    cfg = make_microduck_carry_env_cfg()
    assert "head_pose_bias" in cfg.rewards
    assert cfg.rewards["head_pose_bias"].params["tau_s"] >= 1.0


def test_latch_is_not_registered():
    """Phase 2 carries a toy it is GIVEN; it must not be able to re-acquire one.

    A live ``update_grasp_latch`` would let a policy that lost the toy silently
    grab it again, masking a failure to carry.
    """
    cfg = make_microduck_carry_env_cfg()
    assert "grasp_latch" not in cfg.events
    funcs = [getattr(t, "func", None) for t in cfg.events.values()]
    from mjlab_microduck.tasks.mdp import update_grasp_latch

    assert update_grasp_latch not in funcs


def test_attach_event_runs_after_the_robot_reset_events():
    """Events fire in dict insertion order and the toy is placed off the robot's
    FINAL reset pose, so attach_toy must come after reset_base."""
    cfg = make_microduck_carry_env_cfg()
    order = list(cfg.events.keys())
    assert "attach_toy" in order
    assert order.index("attach_toy") > order.index("reset_base")

    term = cfg.events["attach_toy"]
    assert term.mode == "reset"
    assert term.params["offset_in_head"] == CARRY_OFFSET_IN_HEAD


def test_attach_event_declares_the_eq_data_expansion():
    """``m.eq_data`` ships as ONE shared row. Without the per-world expansion
    declaration every env silently gets env 0's weld transform — which, since the
    transform is a constant here, would look completely fine right up until any
    per-env variation is introduced."""
    from mjlab_microduck.tasks.mdp import attach_toy_to_beak

    assert "eq_data" in set(getattr(attach_toy_to_beak, "model_fields", ())), (
        "attach_toy_to_beak lost @requires_model_fields('eq_data')"
    )


def test_payload_penalty_has_a_costing_sign():
    """AGENTS.md sign convention: self-negating ``*_penalty`` functions return
    <= 0 and therefore take a POSITIVE weight. A negative weight here would
    double-negate into a REWARD for flinging the toy — exactly the behaviour the
    term exists to suppress, and the policy would farm it."""
    cfg = make_microduck_carry_env_cfg()
    assert "payload_violence" in cfg.rewards
    assert cfg.rewards["payload_violence"].weight > 0.0
    assert cfg.rewards["payload_violence"].weight == PAYLOAD_VIOLENCE_WEIGHT
    # It is a guard rail, not an objective: it must not outweigh gait tracking.
    assert cfg.rewards["payload_violence"].weight < cfg.rewards[
        "track_linear_velocity"
    ].weight


def test_actor_stays_blind_to_the_toy():
    """The deployed robot has no toy sensing (that is roadmap Phase 4) and the
    actor obs is a fixed 61 D contract. Toy state is critic-only."""
    cfg = make_microduck_carry_env_cfg()
    actor = cfg.observations["actor"].terms
    critic = cfg.observations["critic"].terms
    for name in ("toy_pos", "toy_vel", "grasp_state"):
        assert name not in actor, f"actor must not see {name}"
        assert name in critic, f"critic should see {name}"


def test_toy_mass_randomization_spans_the_measured_range():
    """The DR must cover the payload range Phase 1 actually measured: 10-80 g.

    Deliberately asserts ABSOLUTE masses rather than re-deriving the expectation
    from ``TOY_MASS_RANGE_KG``. Mutation testing caught the tautological version:
    computing the expected alpha from the same constant the cfg used made the test
    pass for any range at all, including one that never leaves 20-50 g and would
    silently stop training the heavy end that costs 14.5 % of the sole.
    """
    cfg = make_microduck_carry_env_cfg()
    term = cfg.events["randomize_toy_mass"]
    a_lo, a_hi = term.params["alpha_range"]
    # pseudo_inertia takes alpha = ln(mass / nominal) / 2; decode it back.
    lo_kg = TOY_NOMINAL_KG * math.exp(2.0 * a_lo)
    hi_kg = TOY_NOMINAL_KG * math.exp(2.0 * a_hi)
    assert lo_kg == pytest.approx(0.010, abs=1e-4), f"light end is {lo_kg*1e3:.1f} g"
    assert hi_kg == pytest.approx(0.080, abs=1e-4), f"heavy end is {hi_kg*1e3:.1f} g"


def test_flat_terrain_only():
    """Rough terrain claims scene.spec_fn for contact softening, which the weld
    needs. Failing loudly beats silently dropping the weld."""
    with pytest.raises(ValueError, match="flat-terrain only"):
        make_microduck_carry_env_cfg(rough=True)


def test_carry_offset_is_clear_of_the_head_collision_mesh():
    """The measured constant, re-derived against the actual compiled model.

    This is the assertion that survives a robot re-export: if the head mesh
    changes, a hard-coded offset that used to hang just clear of the beak can end
    up buried inside it, and a welded toy fighting a contact injects force into
    the head every single step. Asserting the CONSTANT alone would be vacuous —
    this compiles the scene, places the toy where the weld will hold it, and
    checks the physics.
    """
    from copy import deepcopy

    import mujoco

    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg
    from mjlab_microduck.robot.microduck_constants import (
        MICRODUCK_GRASP_LIFT_ROBOT_CFG,
        MICRODUCK_TOY_CFG,
    )
    from mjlab_microduck.tasks.mdp import make_grasp_weld_spec_fn

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
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/jaw_soft")
    toy_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "toy/toy")
    toy_g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "toy/toy_geom")
    qadr = model.jnt_qposadr[model.body_jntadr[toy_b]]

    x_head = data.xpos[head].copy()
    R_head = data.xmat[head].reshape(3, 3).copy()

    data.qpos[qadr : qadr + 3] = x_head + R_head @ np.asarray(CARRY_OFFSET_IN_HEAD)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R_head.flatten())
    data.qpos[qadr + 3 : qadr + 7] = q
    mujoco.mj_forward(model, data)

    worst = 0.0
    for c in range(data.ncon):
        con = data.contact[c]
        if toy_g in (con.geom1, con.geom2):
            worst = min(worst, float(con.dist))
    assert worst >= -1e-4, (
        f"carried toy interpenetrates the head by {-worst*1e3:.2f} mm — the weld "
        "would fight a contact every step; re-run the offset sweep"
    )




# ── Mechanism tests: instantiate and step the real env ───────────────────────

def _carry_env(num_envs: int = 4, **attach_params):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-Carry-Flat-MicroDuck")
    cfg.scene.num_envs = num_envs
    # Seeded: spawn pose, CoM randomization and pushes are all random, and an
    # unseeded run makes these flaky rather than wrong.
    cfg.seed = 0
    for k, v in attach_params.items():
        cfg.events["attach_toy"].params[k] = v
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def _offset_error(env):
    """Toy position in the head frame, minus the offset the weld should hold."""
    from mjlab.utils.lab_api.math import quat_apply_inverse

    robot, toy = env.scene["robot"], env.scene["toy"]
    head_bid = int(robot.indexing.body_ids[robot.body_names.index("jaw_soft")])
    rel = toy.data.root_link_pos_w - env.sim.data.xpos[:, head_bid]
    in_head = quat_apply_inverse(env.sim.data.xquat[:, head_bid], rel)
    want = torch.tensor(CARRY_OFFSET_IN_HEAD, dtype=in_head.dtype).unsqueeze(0)
    return torch.linalg.norm(in_head - want, dim=-1)


def test_toy_spawns_welded_at_the_beak_and_stays_there():
    """The end-to-end claim of Phase 2's design: the episode STARTS holding.

    Config tests cannot see this. If ``attach_toy_to_beak`` wrote ``eq_data`` but
    never set ``eq_active``, or placed the toy and never welded it, every
    assertion above would still pass while the toy simply fell on the floor at
    step 0 and the task trained "walk near a block".
    """
    env = _carry_env()
    try:
        from mjlab_microduck.tasks.mdp import _grasp_eq_id, _grasp_flag

        env.reset()
        env.sim.forward()

        assert bool(_grasp_flag(env).all()), "toy is not marked held at reset"
        assert bool(
            env.sim.data.eq_active[:, _grasp_eq_id(env)].all()
        ), "weld inactive at reset — the toy is not actually attached"
        assert bool(
            (_offset_error(env) < 1e-3).all()
        ), f"toy misplaced at reset: {_offset_error(env)*1e3} mm"

        # Now MOVE, and check the weld actually holds through motion. Random
        # actions shake the head far harder than zero actions do.
        gen = torch.Generator().manual_seed(0)
        worst = 0.0
        for _ in range(40):
            action = 0.5 * torch.randn(
                env.num_envs, env.action_manager.total_action_dim, generator=gen
            )
            env.step(action)
            worst = max(worst, float(_offset_error(env).max()))
        assert worst < 2e-3, f"toy drifted {worst*1e3:.2f} mm out of the beak"
        assert bool(torch.isfinite(env.sim.data.qpos).all())
    finally:
        env.close()


def test_kinematics_refresh_is_what_makes_the_placement_correct():
    """Guards the reset-staleness trap documented on attach_toy_to_beak.

    mjlab fires reset events BEFORE the forward() at the end of step, and the head
    is not the root body, so its xpos is still the PREVIOUS episode's until FK is
    re-run. Without the refresh the toy is spawned metres from the beak — while
    ``eq_active``, the held flag, finite rewards and NaN guards all still look
    perfectly healthy. That silence is why this needs an explicit test.

    Measured: 0.00006 mm of error with the refresh, 1836 mm without.
    """
    env = _carry_env(refresh_kinematics=False)
    try:
        env.reset()
        env.sim.forward()
        stale_err = float(_offset_error(env).max())
    finally:
        env.close()

    env = _carry_env(refresh_kinematics=True)
    try:
        env.reset()
        env.sim.forward()
        fresh_err = float(_offset_error(env).max())
    finally:
        env.close()

    assert fresh_err < 1e-3, f"refresh path is broken: {fresh_err*1e3:.3f} mm"
    assert stale_err > 0.05, (
        "skipping the kinematics refresh no longer misplaces the toy — either "
        "mjlab now refreshes FK before reset events (in which case simplify "
        f"attach_toy_to_beak) or this test has gone vacuous (err {stale_err*1e3:.3f} mm)"
    )


def test_obs_contract_is_61d_and_eq_data_is_per_world():
    """Two invariants in one env instantiation (each is slow to set up).

    61 D actor: policies are hot-swapped against one shared command buffer, so an
    env that changes the width breaks every other policy in the family.
    Per-world ``eq_data``: the AGENTS.md weld trap.
    """
    env = _carry_env(num_envs=4)
    try:
        obs, _ = env.reset()
        assert obs["actor"].shape[-1] == 61, f"actor obs is {obs['actor'].shape[-1]}D"
        assert env.sim.model.eq_data.shape[0] == env.num_envs, (
            "eq_data was not expanded per-world — every env would share env 0's weld"
        )
    finally:
        env.close()


def test_payload_penalty_is_free_during_normal_motion_and_bites_when_flung():
    """The hinge must actually hinge.

    A plain |a| penalty is the unescapable-tax mistake (a 38 %-of-mass head MUST
    oscillate to walk), so this term has to charge exactly zero below
    ``free_accel`` and bite above it. A regression to a plain magnitude penalty
    would still be <= 0 and still pass the sign test, so that check alone is not
    enough.
    """
    from mjlab_microduck.tasks.mdp import (
        _carry_prev_toy_vel,
        carried_toy_accel_penalty,
    )

    env = _carry_env(num_envs=4)
    try:
        env.reset()
        toy = env.scene["toy"]
        # Past the fresh-reset guard, which zeroes the term for the first steps.
        action = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
        for _ in range(3):
            env.step(action)

        def force_accel(delta_v: float):
            """Set up a known acceleration for the CURRENT step."""
            _carry_prev_toy_vel(env)[:] = toy.data.root_link_lin_vel_w
            _carry_prev_toy_vel(env)[:, 0] -= delta_v
            # The env's own managers already cached this step's acceleration;
            # drop it so the poked buffer is what gets differenced.
            if hasattr(env, "_carry_accel_cache"):
                del env._carry_accel_cache

        # Gentle: a velocity change well under free_accel * dt costs nothing.
        force_accel(0.5 * PAYLOAD_FREE_ACCEL * env.step_dt)
        assert bool(
            (carried_toy_accel_penalty(env) == 0.0).all()
        ), "normal-gait payload motion must be free"

        # Violent: several times the hinge must cost.
        force_accel(5.0 * PAYLOAD_FREE_ACCEL * env.step_dt)
        assert bool(
            (carried_toy_accel_penalty(env) < 0.0).all()
        ), "flinging the toy must cost"
    finally:
        env.close()


def test_payload_diagnostics_are_not_dead():
    """Both metrics must report the REAL acceleration, not a constant zero.

    mjlab computes rewards BEFORE metrics. The first implementation had each of the
    three terms difference one shared previous-velocity buffer, so the penalty
    consumed it and both metrics then measured ``vel - vel == 0`` forever.

    This is not hypothetical — it shipped to the first Carry smoke test, which
    reported ``carried_toy_accel = 0.0000`` and ``carried_toy_grip_force =
    0.38 N``. That force is exactly (mean randomized toy mass) x 9.81, i.e. the
    payload's static weight with the entire dynamic term missing, and it reads like
    a perfectly healthy number. Instrumentation that always returns zero is worse
    than no instrumentation, because the whole point of these two metrics is to
    replace a guessed ``free_accel`` with a measured distribution.
    """
    from mjlab_microduck.tasks.mdp import (
        carried_toy_accel,
        carried_toy_accel_penalty,
        carried_toy_grip_force,
    )

    env = _carry_env(num_envs=4)
    try:
        env.reset()
        gen = torch.Generator().manual_seed(0)
        peak_accel = 0.0
        peak_force = 0.0
        static_weight = 0.0
        for _ in range(25):
            action = 0.8 * torch.randn(
                env.num_envs, env.action_manager.total_action_dim, generator=gen
            )
            env.step(action)
            # Read them the way the env does: rewards first, then metrics. Under
            # the shared-buffer bug this ordering is exactly what zeroed them.
            pen = carried_toy_accel_penalty(env)
            accel = carried_toy_accel(env)
            force = carried_toy_grip_force(env)

            assert bool((pen <= 0.0).all()), "penalty must never pay"
            peak_accel = max(peak_accel, float(accel.max()))
            peak_force = max(peak_force, float(force.max()))

            toy = env.scene["toy"]
            body_mass = env.sim.model.body_mass
            bid = int(toy.indexing.root_body_id)
            mass = body_mass[:, bid] if body_mass.dim() > 1 else body_mass[bid]
            static_weight = float((mass * 9.81).max())

        assert peak_accel > 1.0, (
            f"carried_toy_accel never left zero (peak {peak_accel:.4f} m/s^2) — the "
            "metric is dead and cannot inform PAYLOAD_FREE_ACCEL"
        )
        assert peak_force > static_weight * 1.05, (
            f"carried_toy_grip_force peaked at {peak_force:.4f} N vs a static "
            f"weight of {static_weight:.4f} N — the dynamic term is missing"
        )
    finally:
        env.close()
