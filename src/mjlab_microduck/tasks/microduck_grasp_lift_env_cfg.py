"""Microduck GraspLift task — roadmap Phase 1: grasp a real toy and lift it.

Crouch, close the beak on a small light block on the floor, and stand back up
holding it. This is the first task in the family where the robot actually HOLDS
something: ``ground_pick`` reaches for the floor and then fakes a payload with an
external wrench (``apply_mouth_payload_force``), and nothing is ever grasped.

Why the grasp is a weld
-----------------------
The Microduck has no jaw servo. Its 14 actuated joints are 5+5 leg and 4
neck/head, and the beak geoms are rigidly fixed to the ``jaw_soft`` head body, so
"close the beak" is not an action the policy can take. The grasp is therefore a
latched ``mjEQ_WELD`` equality between the head and the toy, installed inactive at
compile time by ``mdp.make_grasp_weld_spec_fn`` and switched on per-env by
``mdp.update_grasp_latch`` when the beak genuinely closes on the toy: real contact,
mouth within 35mm of the toy centre, mouth pointing down, and low relative speed.
The speed gate is what stops the obvious exploit — a fast head slam that clips the
toy in passing is not a grasp.

Measured geometry (scripts/payload_sweep.py, and a reach scan over the leg+neck
joint ranges with both feet grounded — never assume these, re-measure them if the
robot model is re-exported):
  * mouth_tip bottoms out at z ~= 13mm, at x ~= 68mm in the yaw frame;
    it still reaches the floor out to x ~= 128mm.  -> toy spawns at x = 80 +/- 20mm
  * standing HOME mouth_tip height ~= 214mm.        -> TOY_CARRY_Z
  * toy is a 30x30x24mm block, so it rests centred at z = 12mm.

Payload range
-------------
``scripts/payload_sweep.py`` says servo torque is NOT the binding constraint: even
200 g needs only 26% of the XL330's stall torque at the neck. What binds is
BALANCE — a 80 g toy on the beak drags the whole-body CoM 8.2mm forward, 18% of
the 46mm sole the robot has to keep its weight over (200 g would be 39%). So the
mass range is trained as a curriculum from the safe 10-40 g that ground_pick
assumed, out to 80 g, rather than jumping straight to dog-toy masses.

Phase encoding is ground_pick's segmented profile (in the twist command slot):
    descent [0, 0.375)      1.5 s  stand -> down at the toy
    grasp   [0.375, 0.45)   0.3 s  low, still, latching
    lift    [0.45, 0.80)    1.4 s  back up to standing, holding
    rest    [0.80, 1)       0.8 s  standing with the toy
The latch itself is NOT phase-gated: it fires on physical conditions alone, so the
policy cannot earn a grasp by being in the right place at the right time.

DR / obs / regularization: velocity-parity, inherited from the ground_pick recipe
(itself matched to the velocity env — the recipe with proven transfer).
"""

import math
from copy import deepcopy

# Symmetry — must stay OFF: SYMMETRY_CFG's permutation table predates the 61D obs
# layout, and this task is not left/right symmetric anyway (the toy spawns with
# lateral placement noise).
ENABLE_SYMMETRY = False

# ── Domain randomisation toggles (matched to velocity / ground_pick) ──────────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False
ENABLE_KD_RANDOMIZATION              = False
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True
ENABLE_TOY_MASS_RANDOMIZATION        = True

# ── Ranges (matched to velocity / ground_pick) ───────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003   # ramped to 0.02 via curriculum
HEAD_COM_RANDOMIZATION_RANGE        = 0.003   # ramped to 0.01 via curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)   # unused (kp DR off)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)     # unused (kd DR off)
VELOCITY_PUSH_INTERVAL_S            = (3.0, 6.0)
# Quasi-static gesture -> gentle pushes, same as ground_pick (+/-0.3 knocked it
# over even standing straight).
VELOCITY_PUSH_RANGE                 = (-0.15, 0.15)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# ── Toy geometry / mass (keep in sync with robot/microduck/toy.xml) ──────────
TOY_NOMINAL_KG    = 0.03     # <inertial mass> in toy.xml
TOY_HALF_HEIGHT   = 0.012    # half of the 24mm block height -> resting centre z
TOY_MASS_RANGE_KG = (0.01, 0.08)  # curriculum-ramped; see the payload note above

# ── Toy placement (robot yaw frame, from the measured reach envelope) ────────
TOY_OFFSET_XY     = (0.08, 0.0)
TOY_POS_NOISE_XY  = 0.02

# ── Grasp gate ───────────────────────────────────────────────────────────────
GRASP_RADIUS        = 0.035   # mouth_tip-to-toy-centre distance allowed to latch
GRASP_MIN_ALIGNMENT = 0.3     # cos(mouth axis, world -Z); 1.0 = straight down
GRASP_MAX_REL_SPEED = 0.35    # m/s; above this it is a slap, not a grasp

# ── Carry target ─────────────────────────────────────────────────────────────
# Standing mouth_tip height is 214mm (measured). The toy is welded wherever it was
# caught, so it hangs somewhere within GRASP_RADIUS of the beak; the target sits a
# little under the mouth and the std is deliberately loose enough to cover that
# whole ball. Tightening it would punish a good grasp for an unescapable offset.
TOY_CARRY_Z   = 0.20
TOY_CARRY_STD = 0.06

# ── Phase profile (ground_pick's segmented gating; see module docstring) ─────
GL_PERIOD   = 4.0
DESCENT_END = 0.375
HOLD_END    = 0.45
RISE_END    = 0.80

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_GRASP_LIFT_ROBOT_CFG,
    MICRODUCK_TOY_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_grasp_lift_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the Microduck GraspLift environment configuration.

    Flat terrain only: a toy resting on procedurally generated rough terrain is a
    different (and much later) task — it would not even spawn at a known height.
    """

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # Head-on-GROUND impact: discourages slamming the head into the floor during
    # the approach. Note this is head-vs-terrain only — touching the TOY is the
    # point of the task and is sensed separately below.
    head_impact_cfg = ContactSensorCfg(
        name="head_impact_contact",
        primary=ContactMatch(mode="subtree", pattern="neck", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    # Head-on-TOY contact: the physical precondition of the grasp latch. Without
    # this the "grasp" would be pure proximity, and the policy could earn one by
    # hovering the beak near the toy without ever touching it.
    head_toy_cfg = ContactSensorCfg(
        name="head_toy_contact",
        primary=ContactMatch(mode="subtree", pattern="neck", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="toy", entity="toy"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    # Robot MUST stay the first entity: set_random_ground_state and the base reset
    # events write robot root state at qpos[:, 0:7].
    cfg.scene.entities = {
        "robot": MICRODUCK_GRASP_LIFT_ROBOT_CFG,
        "toy": MICRODUCK_TOY_CFG,
    }
    cfg.scene.sensors = (
        feet_ground_cfg,
        self_collision_cfg,
        head_impact_cfg,
        head_toy_cfg,
    )
    cfg.viewer.body_name = "trunk_base"

    # Install the (inactive) grasp weld. Entities are attached under a
    # "<name>/" prefix, so the bodies are addressed as robot/jaw_soft and toy/toy.
    cfg.scene.spec_fn = microduck_mdp.make_grasp_weld_spec_fn(
        head_body="robot/jaw_soft", toy_body="toy/toy"
    )

    # Contact headroom for the toy (toy-terrain + toy-robot on top of the
    # full-collision robot's own budget), and constraint headroom for the weld,
    # which adds 6 rows to every world that is currently holding the toy.
    cfg.sim.nconmax = 50
    cfg.sim.njmax = 120

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0
    # No NeckOffsetJointPositionAction — the head IS the task effector here.

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",  # gait-conditioned; replaced by the phased return-pose terms
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: reach the toy (descent + grasp window) ───────────────────────
    # Distance to the TOY, not to the floor like ground_pick: the target is the
    # object, so the term keeps paying once the toy leaves the ground and never
    # fights head_impact_penalty. std = 0.10 gives gradient from the standing pose.
    cfg.rewards["mouth_toy_proximity"] = RewardTermCfg(
        func=microduck_mdp.mouth_toy_proximity_phased,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=["mouth_tip"]),
            "asset_name": "toy",
            "std": 0.10,
            "command_name": "twist",
            "descent_end": DESCENT_END,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )

    # Beak pointing down. Also the latch's orientation gate, so reward and gate
    # agree on what "mouth down" means.
    cfg.rewards["mouth_perpendicular"] = RewardTermCfg(
        func=microduck_mdp.mouth_perpendicular_phased,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=["mouth_tip"]),
            "command_name": "twist",
            "descent_end": DESCENT_END,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )

    # ── Rewards: the grasp itself ─────────────────────────────────────────────
    # One-shot impulse on the latch step. Deliberately NOT a per-step "is holding"
    # payout starting at the grab: that pays more the earlier (and so the more
    # violently) the toy is caught. Weight 30 against a one-step 1.0 makes a
    # successful grasp worth about as much as a second of the standing stack.
    cfg.rewards["grasp_engage"] = RewardTermCfg(
        func=microduck_mdp.grasp_engage_bonus,
        weight=30.0,
    )

    # Still holding it, ramped in over the lift. Zero for the whole descent, so an
    # early grab collects nothing extra — this term prices DROPPING, not grabbing.
    cfg.rewards["grasp_held"] = RewardTermCfg(
        func=microduck_mdp.grasp_held_reward,
        weight=3.0,
        params={"command_name": "twist", "hold_end": HOLD_END, "rise_end": RISE_END},
    )

    # Potential-based lift: pays the toy's per-step height CHANGE while held, so
    # raising pays, holding pays zero and lowering refunds. A height-LEVEL reward
    # would be a jackpot the policy would reach as fast as it could and camp on.
    # Weight 200 x a <=0.02 m/step delta caps the per-step payout at 4.
    cfg.rewards["toy_lift"] = RewardTermCfg(
        func=microduck_mdp.toy_lift_progress,
        weight=200.0,
        params={
            "asset_name": "toy",
            "command_name": "twist",
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
            "max_step": 0.02,
        },
    )

    # Terminal objective: standing at rest with the toy up at beak height.
    cfg.rewards["toy_carried"] = RewardTermCfg(
        func=microduck_mdp.toy_carried_height,
        weight=4.0,
        params={
            "asset_name": "toy",
            "command_name": "twist",
            "target_height": TOY_CARRY_Z,
            "std": TOY_CARRY_STD,
            "rise_end": RISE_END,
        },
    )

    # Prices the "just nudge it" failure the roadmap flags for real hardware: a
    # beak that swipes the toy across the floor is strictly worse than one that
    # closes on it. Zero once the toy is actually held, so carrying it is free.
    cfg.rewards["toy_knocked_away"] = RewardTermCfg(
        func=microduck_mdp.toy_knocked_away_penalty,
        weight=-4.0,
        params={"asset_name": "toy", "deadzone": 0.01, "max_cost": 0.2},
    )

    # ── Rewards: return to standing (ground_pick's phased stack) ──────────────
    cfg.rewards["return_pose_legs"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_pose_phased,
        weight=6.0,
        params={
            "std": 0.3,
            "command_name": "twist",
            "joint_indices": _LEG_JOINTS,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )

    # Neck/head: tight std to prevent backward overshoot and head-body collision
    # (head geoms have no collision mesh, so self_collisions cannot catch it — the
    # pose reward is the only guard).
    cfg.rewards["return_pose_neck"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_pose_phased,
        weight=6.0,
        params={
            "std": 0.15,
            "command_name": "twist",
            "joint_indices": _NECK_JOINTS,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )

    # Trunk vertical during the stand-up only: returning to pose does not by itself
    # guarantee dynamic balance while extending. Gated on the rise so it does not
    # fight the forward lean of the approach.
    cfg.rewards["return_upright"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_upright_phased,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.4,
            "command_name": "twist",
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )

    # Anti-dive: brakes the neck during descent+hold only (gate 0 on the rise, so it
    # does not throttle the lift).
    cfg.rewards["neck_vel_descent"] = RewardTermCfg(
        func=microduck_mdp.neck_vel_descent_penalty,
        weight=-0.1,
        params={
            "command_name": "twist",
            "joint_indices": _NECK_JOINTS,
            "hold_end": HOLD_END,
        },
    )

    # ── Rewards: stability ────────────────────────────────────────────────────
    # Upright: low weight — the robot has to lean forward to reach the toy.
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 0.2

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["soft_landing"].weight = -1e-5

    # Both feet stay down through the pick.
    cfg.rewards["feet_grounded"] = RewardTermCfg(
        func=microduck_mdp.feet_grounded_reward,
        weight=3.0,
        params={"sensor_name": feet_ground_cfg.name},
    )

    # Feet FLAT. feet_grounded only sees contact, so a foot pivoting onto its edge
    # still counts; this projects gravity into the foot site frame to forbid it.
    cfg.rewards["feet_flat"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=["left_foot", "right_foot"]),
        },
    )

    # ── Rewards: regularisation (heavier than velocity — slow careful reaching) ─
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.8)
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=-1.0
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-5e-3
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Head-into-FLOOR impact. Weaker than ground_pick's -2.0: there the mouth had to
    # hover without touching anything, here it has to descend far enough to press on
    # a 24mm block sitting on that floor, and too harsh a floor penalty makes the
    # policy stop short of the toy.
    cfg.rewards["head_impact_penalty"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=-1.0,
        params={"sensor_name": head_impact_cfg.name, "threshold": 1.0},
    )

    # ── Observations (unified 61D actor layout, toy-BLIND) ────────────────────
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    # No terrain-height sensor in this env (flat only).
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    # Sensor delay — matches velocity / ground_pick.
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 3
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 3
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # 1-ctrl-step lag on joint_vel (Dynamixel moving-average, see velocity env).
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # Command obs slots — unified layout [twist(3), head(4), body(6)]; the twist
    # slot carries the phase, head/body stay zero-padded so the runtime can
    # hot-swap this ONNX against the walking policy with one command buffer.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # CRITIC-ONLY toy state (asymmetric actor-critic). The actor stays blind: the
    # real robot has no toy sensing until roadmap Phase 4 (perception), so
    # robustness to placement error has to come from TOY_POS_NOISE_XY, not from an
    # observation that will not exist on hardware.
    cfg.observations["critic"].terms["toy_position"] = ObservationTermCfg(
        func=microduck_mdp.toy_pos_in_base, params={"asset_name": "toy"},
    )
    cfg.observations["critic"].terms["toy_velocity"] = ObservationTermCfg(
        func=microduck_mdp.toy_vel_in_base, params={"asset_name": "toy"},
    )
    cfg.observations["critic"].terms["grasp_state"] = ObservationTermCfg(
        func=microduck_mdp.grasp_state_obs,
    )

    # ── Command: cyclic phase encoding ────────────────────────────────────────
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{
            **vars(command),
            "class_type": microduck_mdp.GroundPickPhaseCommand,
            "period": GL_PERIOD,
        }
    )

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)

    # Release the weld and clear grasp bookkeeping. ALSO declares the per-world
    # eq_data expansion for the whole task (see mdp.reset_grasp) — without it every
    # env would share env 0's weld pose.
    cfg.events["reset_grasp"] = EventTermCfg(
        func=microduck_mdp.reset_grasp,
        mode="reset",
        params={"asset_name": "toy"},
    )

    # Toy placement. MUST come after reset_base (events run in dict insertion
    # order) because the spawn is derived from the final robot pose.
    cfg.events["reset_toy"] = EventTermCfg(
        func=microduck_mdp.reset_toy_on_ground,
        mode="reset",
        params={
            "offset": TOY_OFFSET_XY,
            "noise_xy": TOY_POS_NOISE_XY,
            "yaw_noise": math.pi,
            "toy_half_height": TOY_HALF_HEIGHT,
            "asset_name": "toy",
        },
    )

    # The grasp latch. "step" mode fires after sim.forward(), so site/body poses are
    # fresh and the weld switched on here takes effect on the next physics step.
    cfg.events["grasp_latch"] = EventTermCfg(
        func=microduck_mdp.update_grasp_latch,
        mode="step",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["jaw_soft"], site_names=["mouth_tip"]
            ),
            "asset_name": "toy",
            "sensor_name": head_toy_cfg.name,
            "radius": GRASP_RADIUS,
            "min_alignment": GRASP_MIN_ALIGNMENT,
            "max_rel_speed": GRASP_MAX_REL_SPEED,
        },
    )

    if ENABLE_TOY_MASS_RANDOMIZATION:
        # pseudo_inertia (not body_mass): it scales mass AND inertia together, which
        # is the physically consistent "same block, different density" change.
        # alpha = ln(scale)/2, with scale relative to toy.xml's 30 g nominal.
        # Startup mode + curriculum, matching how the other mass DR is applied.
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

    if ENABLE_VELOCITY_PUSHES:
        interval = (2.0, 4.0) if play else VELOCITY_PUSH_INTERVAL_S
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # ── Terrain: flat only ────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum ────────────────────────────────────────────────────────────
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # Action-rate ramp: light while the gross reach-and-lift motion is still being
    # discovered, then clamped down hard for transfer. Any attempt-tax applied
    # while a hard skill is being explored makes "do nothing" win.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,        "weight": -0.8},
                {"step": 250 * 24, "weight": -1.5},
                {"step": 500 * 24, "weight": -2.0},
            ],
        },
    )

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                    {"step": 2000 * 24, "range": 0.02},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckGraspLiftRlCfg = RslRlOnPolicyRunnerCfg(
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
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="grasp_lift",
    run_name="grasp_lift",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,
)
