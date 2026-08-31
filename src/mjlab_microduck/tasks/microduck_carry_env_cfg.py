"""Microduck Carry task — roadmap Phase 2: walk while carrying a toy.

Phase 1 (``GraspLift``) taught the robot to pick a block up off the floor. This
task is the next skill: locomote under velocity commands while that block is held
at the beak, with the payload's mass dragging the whole-body CoM forward and
swinging on a head that is already ~38 % of body mass.

Why this is NOT built on ground_pick / grasp_lift
--------------------------------------------------
Phase 1 encodes its pick phase in the TWIST COMMAND SLOT as
``[cos(2*pi*phi), sin(2*pi*phi), 0]`` (``GroundPickPhaseCommand``). Phase 2 needs
that same slot for real velocity commands, and the 61 D observation contract is
shared across the whole policy family so the runtime can hot-swap ONNX policies
against one command buffer — a slot cannot be added, and an unused slot must be
zero-padded rather than deleted. The two uses of the twist slot are therefore
mutually exclusive, and Phase 2 cannot reuse Phase 1's phase machinery at all.

So this builds on ``make_microduck_velocity_env_cfg``: Phase 2 is the walking
recipe plus a payload, and it should inherit the gait rewards, the DR stack, the
obs noise/delay model and the NaN guards rather than the phased pick stack.

The toy is spawned ALREADY WELDED (``mdp.attach_toy_to_beak``) and
``mdp.update_grasp_latch`` is deliberately NOT registered: Phase 2 is about
carrying, not acquiring, and leaving the latch on would let a policy silently
re-grab a toy it lost and mask a failure to carry.

Perfect-grip upper bound (read this before interpreting a run)
--------------------------------------------------------------
A MuJoCo ``mjEQ_WELD`` does not break. With the weld active from step 0 and no
latch, THE TOY CANNOT BE DROPPED — so this task cannot fail by dropping, and a
100 % "carry rate" is guaranteed by construction rather than earned. That is a
deliberate choice, not an oversight:

  * it isolates the difficulty the roadmap actually flags, the CoM coupling;
  * modelling a breakable grip needs a release-force threshold, and the real
    beak's grip strength is unknown until hardware (roadmap Phase 6). Inventing
    that constant would be exactly the kind of assumption this repo re-measures
    instead of guessing.

Instead, ``carried_toy_grip_force`` logs the force the weld actually supplies, so
when Pancha ships, a measured beak grip strength can be compared against a real
distribution — and a breakable-grip variant becomes a small additive change.

Measured geometry (scripts under ``_probe_*``; CPU, no GPU — RE-MEASURE these if
the robot model is re-exported, never carry them across model revisions):
  * In the head (``jaw_soft``) frame, +X is world UP and -Z is world FORWARD.
  * ``mouth_tip`` sits at (-8.09, 0, -77.74) mm in that frame, i.e. 77.7 mm
    forward of and 8.1 mm below the head body origin.
  * The toy is a 30x30x24 mm block (half-extents 15/15/12 mm).
  * Sweeping the toy centre DOWN from ``mouth_tip``, its interpenetration with the
    head collision mesh falls to exactly zero at 25 mm (24 mm still overlaps by
    0.35 mm; centring it on ``mouth_tip`` overlaps by 12.5 mm). 25 mm is therefore
    the carry analogue of where Phase 1's latch fired, since Phase 1 welded on
    CONTACT — i.e. at this same just-touching boundary.
    -> CARRY_OFFSET_IN_HEAD = (-0.033093, 0.0, -0.077738) m

CoM coupling, measured at that carry offset (home standing pose):
    toy mass    CoM shift    % of the 46 mm sole
      10 g        0.91 mm         2.0 %
      20 g        1.80 mm         3.9 %
      30 g        2.67 mm         5.8 %
      50 g        4.33 mm         9.4 %
      80 g        6.67 mm        14.5 %
Slightly GENTLER than Phase 1's measured 8.2 mm at 80 g (18 % of the sole),
because the toy hangs 25 mm below the beak tip — closer to the body — rather than
wherever the Phase 1 latch happened to catch it. The static shift is not the hard
part though: walking turns it into a dynamic disturbance, which is what this task
has to learn to reject.

Head tracking under a payload — already handled by the base recipe
------------------------------------------------------------------
AGENTS.md records that a tight INSTANTANEOUS head-tracking std once taxed walking
so hard the policy stood still (a 38 %-of-mass head must oscillate to walk), and
that the fix is to price only the escapable DC part: L1 on a 1 s EMA. The
velocity recipe already implements exactly that (``head_pose_bias_penalty``,
ramped by the ``head_pose_bias_weight`` curriculum), so building on it inherits
the correct treatment of the payload's steady droop for free. Do NOT "fix" a
head-droop symptom in this task by tightening ``head_pose_tracking``.
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers import (
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
)
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_GRASP_LIFT_ROBOT_CFG,
    MICRODUCK_TOY_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

# Symmetry — must stay OFF, same reasoning as GraspLift: SYMMETRY_CFG's
# permutation table predates the 61 D obs layout.
ENABLE_SYMMETRY = False

ENABLE_TOY_MASS_RANDOMIZATION = True
ENABLE_PAYLOAD_VIOLENCE_PENALTY = True

# ── Toy geometry / mass (keep in sync with robot/microduck/toy.xml) ──────────
TOY_NOMINAL_KG = 0.03
TOY_MASS_RANGE_KG = (0.01, 0.08)  # mirrors Phase 1's measured payload range

# ── Carry pose (MEASURED — see the module docstring) ─────────────────────────
# Toy body origin in the head (jaw_soft) frame; head +X is world up, -Z forward.
# 25 mm below mouth_tip is the first offset with zero interpenetration.
CARRY_OFFSET_IN_HEAD = (-0.033093, 0.0, -0.077738)

# ── Payload violence guard ───────────────────────────────────────────────────
# A BACKSTOP, not a measured threshold. Under a permanent weld nothing else stops
# a head-flinging gait that would rip the toy out of a real beak, but a head that
# must oscillate to walk means the toy must accelerate — so charge nothing below
# free_accel and only price the violent tail above it. The carried_toy_accel
# metric logs the real distribution so the first run MEASURES the right value.
PAYLOAD_FREE_ACCEL = 30.0  # m/s^2, ~3 g
PAYLOAD_MAX_ACCEL = 150.0
PAYLOAD_VIOLENCE_WEIGHT = 0.02  # small: a guard rail, not an objective


def make_microduck_carry_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the Microduck carry-while-walking environment configuration."""
    if rough:
        # Not a fundamental limit, but two things collide and neither has been
        # measured: the velocity recipe claims scene.spec_fn for
        # _soften_terrain_contacts while the grasp weld needs that same hook, and
        # the toy adds contacts on top of rough terrain's already-raised budget.
        raise ValueError(
            "Carry is flat-terrain only: scene.spec_fn is needed for the grasp "
            "weld, but the rough velocity recipe uses it for _soften_terrain_contacts."
        )

    # ── Base config: the WALKING recipe, not the pick stack ───────────────────
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)

    # Robot MUST stay the first entity: set_random_ground_state and the base reset
    # events write robot root state at qpos[:, 0:7].
    #
    # Allcollisions (the GraspLift robot), not the velocity recipe's walk model:
    # the carry offset above was measured against THIS head collision mesh, and a
    # carried block needs the body collisions to interact with honestly. It also
    # keeps Phase 1 and Phase 2 on one robot model.
    cfg.scene.entities = {
        "robot": MICRODUCK_GRASP_LIFT_ROBOT_CFG,
        "toy": MICRODUCK_TOY_CFG,
    }

    # Install the (compile-time inactive) weld. Entities attach under a "<name>/"
    # prefix, so the bodies are robot/jaw_soft and toy/toy. attach_toy_to_beak
    # switches it on per-env at reset.
    cfg.scene.spec_fn = microduck_mdp.make_grasp_weld_spec_fn(
        head_body="robot/jaw_soft", toy_body="toy/toy"
    )

    # Contact headroom for the toy and constraint headroom for the weld, which
    # adds 6 rows to EVERY world here (unlike Phase 1, where only latched envs
    # carried it) — the toy is welded from step 0 in every env.
    cfg.sim.nconmax = 50
    cfg.sim.njmax = 120

    cfg.viewer.body_name = "trunk_base"

    # ── Events ────────────────────────────────────────────────────────────────
    # Spawn the toy already welded to the beak. MUST come after reset_base /
    # reset_joints (events run in dict insertion order) because the toy is placed
    # from the robot's final reset pose — and note attach_toy_to_beak itself
    # refreshes forward kinematics, since the head is not the root body and its
    # xpos is otherwise still the previous episode's.
    #
    # This term also carries the eq_data per-world expansion declaration for the
    # whole task (@requires_model_fields("eq_data")): m.eq_data ships as ONE
    # shared row, so without it every env would be welded with env 0's transform.
    cfg.events["attach_toy"] = EventTermCfg(
        func=microduck_mdp.attach_toy_to_beak,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["jaw_soft"]),
            "asset_name": "toy",
            "offset_in_head": CARRY_OFFSET_IN_HEAD,
        },
    )

    # NOTE: mdp.update_grasp_latch is deliberately NOT registered. Phase 2 carries
    # a toy it is given; a live latch would let the policy re-acquire a lost toy
    # and hide a failure to carry.
    assert "grasp_latch" not in cfg.events

    if ENABLE_TOY_MASS_RANDOMIZATION:
        # pseudo_inertia (not body_mass): scales mass AND inertia together, the
        # physically consistent "same block, different density" change.
        # alpha = ln(scale)/2, scale relative to toy.xml's 30 g nominal.
        _lo, _hi = TOY_MASS_RANGE_KG
        cfg.events["randomize_toy_mass"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("toy", body_names=("toy",)),
                "alpha_range": (
                    math.log(_lo / TOY_NOMINAL_KG) / 2.0,
                    math.log(_hi / TOY_NOMINAL_KG) / 2.0,
                ),
            },
        )

    # ── Rewards ───────────────────────────────────────────────────────────────
    # The gait stack is inherited wholesale from the velocity recipe; the payload
    # is a disturbance those terms must now reject, not a new objective. The only
    # addition is the anti-flinging guard rail.
    #
    # Self-negating penalty (returns <= 0) -> POSITIVE weight. A negative weight
    # would double-negate into a reward for flinging the toy (AGENTS.md sign
    # convention); the test suite asserts this sign.
    if ENABLE_PAYLOAD_VIOLENCE_PENALTY:
        cfg.rewards["payload_violence"] = RewardTermCfg(
            func=microduck_mdp.carried_toy_accel_penalty,
            weight=PAYLOAD_VIOLENCE_WEIGHT,
            params={
                "asset_name": "toy",
                "free_accel": PAYLOAD_FREE_ACCEL,
                "max_accel": PAYLOAD_MAX_ACCEL,
            },
        )

    # ── Observations: the actor stays BLIND to the toy ────────────────────────
    # The deployed Microduck has no toy sensing (that is roadmap Phase 4), and the
    # 61 D actor contract is fixed anyway. The critic may see it.
    cfg.observations["critic"].terms["toy_pos"] = ObservationTermCfg(
        func=microduck_mdp.toy_pos_in_base, params={"asset_name": "toy"}
    )
    cfg.observations["critic"].terms["toy_vel"] = ObservationTermCfg(
        func=microduck_mdp.toy_vel_in_base, params={"asset_name": "toy"}
    )
    cfg.observations["critic"].terms["grasp_state"] = ObservationTermCfg(
        func=microduck_mdp.grasp_state_obs
    )

    # ── Metrics (diagnostics, no gradient) ────────────────────────────────────
    # These are how a perfect-grip sim run still says something about hardware,
    # and how PAYLOAD_FREE_ACCEL stops being a guess after the first run.
    # Metrics have no weight, so unlike a weight-0 reward term they log the real
    # value rather than a constant 0.
    cfg.metrics["carried_toy_accel"] = MetricsTermCfg(
        func=microduck_mdp.carried_toy_accel, params={"asset_name": "toy"}
    )
    cfg.metrics["carried_toy_grip_force"] = MetricsTermCfg(
        func=microduck_mdp.carried_toy_grip_force, params={"asset_name": "toy"}
    )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────
# Mirrors the velocity runner (this is the walking recipe with a payload), with
# its own experiment_name so runs do not land in the walking project's history.

MicroduckCarryRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    # deepcopy, not a shared reference: MicroduckRlCfg.algorithm is a mutable
    # dataclass, and aliasing it would make any later tweak here silently retune
    # the walking task too.
    algorithm=deepcopy(MicroduckRlCfg.algorithm),
    wandb_project="mjlab_microduck",
    experiment_name="carry",
    run_name="carry",
    save_interval=250,
    num_steps_per_env=24,
    # Gaits need the long budget (AGENTS.md: 4000-6000 for gaits and
    # curriculum-heavy tasks), not an episodic trick's ~1000.
    max_iterations=10_000,
)
