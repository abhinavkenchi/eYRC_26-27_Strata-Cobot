#!/usr/bin/env python3

'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script implements the Manipulation Node for Task 3B of StrataCobot (SC) Theme (eYRC 2026-27):
*  Prioritizes the eBot package when it is available at handover, sorts four ores
*  into their matching rover containers, and uses force-checked pick/place flows.
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          5847
# Author List:      Abhinav Kenchi
# Filename:         task3B_manipulation.py
#  Functions:        quat_to_rot_matrix, compute_tool_pointing_error, rotation_from_quaternion,
#                    rotation_vector, ur7e_tcp_transform, solve_ur7e_ik,
#                    Task3BManipulation.__init__,
#                    Task3BManipulation.tcp_pose_cb, Task3BManipulation.arm_status_cb,
#                    Task3BManipulation.arm_status_detail_cb,
#                    Task3BManipulation.force_cb, Task3BManipulation.joint_states_cb,
#                    Task3BManipulation.publish_zero_twist, Task3BManipulation.publish_zero_joint,
#                    Task3BManipulation.compute_orientation_correction,
#                    Task3BManipulation.limit_velocity_change, Task3BManipulation.publish_twist,
#                    Task3BManipulation.request_controller_switch,
#                    Task3BManipulation.poll_controller_states, Task3BManipulation.set_magnet,
#                   Task3BManipulation.drive_tcp_towards, Task3BManipulation.fault,
#                   Task3BManipulation.control_loop, main
# Nodes:            task3b_manipulation
#                   Publishing Topics  - [ /delta_twist_cmds, /delta_joint_cmds ]
#                   Subscribing Topics - [ /tcp_pose_raw, /joint_states, /arm_status,
#                                          /arm_status_detail, /ur7e/end_effector/force_magnitude ]


################### IMPORT MODULES #######################

import math
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
import tf2_ros
from controller_manager_msgs.srv import ListControllers, SwitchController
from control_msgs.msg import JointJog
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Bool, Float32, Int32
from std_srvs.srv import SetBool


##################### TASK CONSTANTS #######################

BASE_FRAME = 'base_link'
TWIST_CONTROLLER = 'delta_twist_controller'
JOINT_CONTROLLER = 'delta_joint_controller'

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

# Joint origins and limits from ur_description/urdf/ur7e_parameters.xacro and
# ur_description/urdf/ur_macro.xacro. The fixed tool0-to-tcp offset matches
# /tcp_pose_raw, as checked against live joint-state feedback.
UR7E_JOINT_ORIGINS = (
    ((0.0, 0.0, 0.1625), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0)),
    ((-0.425, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((-0.3922, 0.0, 0.1333), (0.0, 0.0, 0.0)),
    ((0.0, -0.0997, 0.0), (math.pi / 2.0, 0.0, 0.0)),
    ((0.0, 0.0996, 0.0), (math.pi / 2.0, math.pi, math.pi)),
)
UR7E_JOINT_LOWER_LIMITS = np.array([
    -2.0 * math.pi, -2.0 * math.pi, -2.792526803190927,
    -2.0 * math.pi, -2.0 * math.pi, -2.0 * math.pi,
])
UR7E_JOINT_UPPER_LIMITS = -UR7E_JOINT_LOWER_LIMITS
UR7E_TCP_LENGTH = 0.02041
IK_ORIENTATION_LENGTH_SCALE = 0.20
IK_POSITION_TOLERANCE = 0.001
IK_ORIENTATION_TOLERANCE = 0.005
IK_MIN_SINGULAR_VALUE = 0.03
IK_PREFERRED_SINGULAR_VALUE = 0.08

# Joint-space unfold target: safely clears table, opens elbow, and points tool downward
TARGET_UNFOLD_JOINTS: List[float] = [-2.220, -1.250, -1.500, -1.900, 1.571, 1.571]

# Mid-corridor waypoint (singularity-free bridge between conveyor and cargo containers)
WP_MID_CORRIDOR = (-0.6000, 0.0000, 0.3800)
LIFT_Z = 0.3800

# Reachable release point within the hopper opening, transformed into base_link.
CRUSHER_HOPPER_TARGET = (-0.820, -0.006, 0.330)
# Keep the high-clearance IK target 40 mm closer to the arm than the release point.
CRUSHER_APPROACH_X_OFFSET = 0.040
CRUSHER_APPROACH_SEED_JOINTS = [3.321, -2.282, -0.820, -1.618, 1.571, 0.810]
CONTAINER_RELEASE_Z = 0.3300

# Rover cargo container drop-off target coordinates in base_link (CAD verified center points)
CONTAINER_TARGETS: Dict[str, Tuple[float, float, float]] = {
    'vanadinite':    (-0.5900, 0.6200, CONTAINER_RELEASE_Z),   # Top compartment (Bin 1)
    'malachite':  (-0.2000, 0.6200, CONTAINER_RELEASE_Z),   # Middle compartment (Bin 2)
    'azurite': (-0.1000, 0.6200, CONTAINER_RELEASE_Z),   # Bottom compartment (Bin 3)
}

# Control loop timing
CONTROL_RATE_HZ = 50.0
CONTROL_PERIOD = 1.0 / CONTROL_RATE_HZ
FEEDBACK_TIMEOUT_SEC = 0.5
ODOMETRY_TIMEOUT_SEC = 1.0
PACKAGE_TF_TIMEOUT_SEC = 5.0
HANDOVER_POSITION_TOLERANCE_M = 0.25
HANDOVER_YAW_TOLERANCE_RAD = 0.30
HANDOVER_MAX_LINEAR_SPEED_MPS = 0.12
HANDOVER_MAX_ANGULAR_SPEED_RPS = 0.12
HANDOVER_SETTLE_DURATION_SEC = 0.10  # s - wait for eBot to be fully settled at dock before handover

# Motion tolerances
POSITION_TOLERANCE = 0.005
TRANSIT_TOLERANCE = 0.055           # Loosened for smooth corridor fly-through
JOINT_TOLERANCE = 0.020
PACKAGE_IK_JOINT_TOLERANCE = 0.010  # Eliminates slow asymptotic crawl during package alignment

# Velocity and acceleration limits for fast but controlled transit
MAX_LINEAR_SPEED = 0.245
MAX_LINEAR_ACCELERATION = 0.90
MAX_ANGULAR_ACCELERATION = 1.0
JOINT_UNFOLD_SPEED = 0.600
CONTAINER_SPEED = 0.245
PACKAGE_TRANSFER_SPEED = 0.245
CONTAINER_DESCENT_SPEED = 0.160
DESCEND_SPEED = 0.150
SEATING_SPEED = 0.010
LIFT_SPEED = 0.220


# Force thresholds and timing constants
FORCE_CONTACT_THRESHOLD = 0.30      # N - initial physical contact detection
FORCE_GRASP_VERIFY = 0.20           # N - minimum force confirming magnetic hold (prevents false retry retreats)
FORCE_HOLD_TARGET = 0.60            # N - target seating contact force
FORCE_CONTACT_SUSTAIN_TIME = 0.02   # s - brief confirmation against force-sensor spikes
MAGNET_SETTLE_TIME = 0.02           # s - short coil response interval
PICKUP_SURFACE_OFFSET = 0.001       # m - offset above estimated ore surface (1mm for immediate contact)
MAX_CONTACT_EXTRA_DESCENT = 0.025   # m - max travel searching for contact
GRASP_SETTLE_TIME = 0.02            # s - short dwell before force verification
LIFT_VERIFY_TIME = 0.02             # s - short dwell at lift apex before force verification
FORCE_RELEASED = 0.60               # N - force threshold indicating successful ore drop
RELEASE_SETTLE_TIME = 0.02          # s - short interval for magnet release to register


##################### HELPER FUNCTIONS #######################

def quat_to_rot_matrix(q: List[float]) -> List[List[float]]:
    '''
    Description:    Converts a quaternion [x, y, z, w] to a 3x3 rotation matrix.

    Args:
        q (List[float]): Quaternion [x, y, z, w].

    Returns:
        List[List[float]]: 3x3 rotation matrix.
    '''
    x, y, z, w = q
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)]
    ]


def compute_tool_pointing_error(r_curr: List[List[float]], target_axis: Optional[List[float]] = None) -> List[float]:
    '''
    Description:    Computes angular velocity error vector to align the tool's physical
                    pointing axis (Z_tcp, Column 2 of rotation matrix) with target_axis
                    in base_link (straight down [0, 0, -1]).

    Args:
        r_curr (List[List[float]]): 3x3 rotation matrix of current tool orientation.
        target_axis (List[float], optional): Desired pointing direction. Defaults to [0.0, 0.0, -1.0].

    Returns:
        List[float]: Angular error vector [wx, wy, wz].
    '''
    if target_axis is None:
        target_axis = [0.0, 0.0, -1.0]

    # Tool's physical longitudinal pointing axis in base_link
    a = [r_curr[0][2], r_curr[1][2], r_curr[2][2]]
    b = target_axis

    cx = a[1] * b[2] - a[2] * b[1]
    cy = a[2] * b[0] - a[0] * b[2]
    cz = a[0] * b[1] - a[1] * b[0]

    c_norm = math.sqrt(cx * cx + cy * cy + cz * cz)
    dot_ab = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    theta = math.atan2(c_norm, dot_ab)

    if c_norm < 1e-5:
        return [0.0, 0.0, 0.0]

    k = theta / c_norm
    return [k * cx, k * cy, k * cz]


def rotation_from_quaternion(quaternion: List[float]) -> np.ndarray:
    """Return a validated rotation matrix from an [x, y, z, w] quaternion."""
    values = np.asarray(quaternion, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError('TCP quaternion must contain four finite values.')
    norm = float(np.linalg.norm(values))
    if norm < 1e-9:
        raise ValueError('TCP quaternion has zero magnitude.')
    values /= norm
    return np.asarray(quat_to_rot_matrix(values.tolist()), dtype=float)


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Return the axis-angle rotation vector for a 3x3 rotation matrix."""
    cosine = max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) * 0.5))
    angle = math.acos(cosine)
    if angle < 1e-7:
        return 0.5 * np.array([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ])
    if math.pi - angle < 1e-5:
        axis = np.zeros(3)
        largest_diagonal = int(np.argmax(np.diag(rotation)))
        if largest_diagonal == 0:
            axis[0] = math.sqrt(max(0.0, (rotation[0, 0] + 1.0) * 0.5))
            divisor = max(4.0 * axis[0], 1e-9)
            axis[1] = (rotation[0, 1] + rotation[1, 0]) / divisor
            axis[2] = (rotation[0, 2] + rotation[2, 0]) / divisor
        elif largest_diagonal == 1:
            axis[1] = math.sqrt(max(0.0, (rotation[1, 1] + 1.0) * 0.5))
            divisor = max(4.0 * axis[1], 1e-9)
            axis[0] = (rotation[0, 1] + rotation[1, 0]) / divisor
            axis[2] = (rotation[1, 2] + rotation[2, 1]) / divisor
        else:
            axis[2] = math.sqrt(max(0.0, (rotation[2, 2] + 1.0) * 0.5))
            divisor = max(4.0 * axis[2], 1e-9)
            axis[0] = (rotation[0, 2] + rotation[2, 0]) / divisor
            axis[1] = (rotation[1, 2] + rotation[2, 1]) / divisor
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-9:
            raise ValueError('Could not determine rotation axis near pi radians.')
        return angle * axis / axis_norm

    scale = angle / (2.0 * math.sin(angle))
    return scale * np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])


def _rotation_matrix_from_rpy(rpy: Tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def ur7e_tcp_transform(joint_values: List[float]) -> np.ndarray:
    """Forward kinematics for base_link -> tcp using the repository UR7e model."""
    joints = np.asarray(joint_values, dtype=float)
    if joints.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(joints)):
        raise ValueError('UR7e FK requires six finite joint angles in JOINT_NAMES order.')

    transform = np.eye(4)
    for angle, (xyz, rpy) in zip(joints, UR7E_JOINT_ORIGINS):
        origin = np.eye(4)
        origin[:3, :3] = _rotation_matrix_from_rpy(rpy)
        origin[:3, 3] = xyz
        joint_rotation = np.eye(4)
        joint_rotation[:3, :3] = np.array([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        transform = transform @ origin @ joint_rotation

    for xyz, rpy in (
        ((0.0, 0.0, 0.0), (0.0, -math.pi / 2.0, -math.pi / 2.0)),
        ((0.0, 0.0, 0.0), (math.pi / 2.0, 0.0, math.pi / 2.0)),
        ((0.0, 0.0, UR7E_TCP_LENGTH), (0.0, 0.0, 0.0)),
    ):
        fixed_transform = np.eye(4)
        fixed_transform[:3, :3] = _rotation_matrix_from_rpy(rpy)
        fixed_transform[:3, 3] = xyz
        transform = transform @ fixed_transform
    return transform


def _ik_residual(
    joints: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    transform = ur7e_tcp_transform(joints.tolist())
    position_error = target_position - transform[:3, 3]
    orientation_error = rotation_vector(target_rotation @ transform[:3, :3].T)
    weighted_error = np.concatenate((
        position_error,
        IK_ORIENTATION_LENGTH_SCALE * orientation_error,
    ))
    return weighted_error, position_error, orientation_error


def _ik_jacobian(joints: np.ndarray, transform: np.ndarray) -> np.ndarray:
    jacobian = np.zeros((6, len(JOINT_NAMES)))
    step_size = 1e-6
    for index in range(len(JOINT_NAMES)):
        perturbed = joints.copy()
        perturbed[index] += step_size
        next_transform = ur7e_tcp_transform(perturbed.tolist())
        jacobian[:3, index] = (
            next_transform[:3, 3] - transform[:3, 3]
        ) / step_size
        jacobian[3:, index] = (
            IK_ORIENTATION_LENGTH_SCALE
            * rotation_vector(next_transform[:3, :3] @ transform[:3, :3].T)
            / step_size
        )
    return jacobian


def solve_ur7e_ik(
    current_joints: List[float],
    target_position: Tuple[float, float, float],
    target_rotation: np.ndarray,
) -> Tuple[Optional[List[float]], str]:
    """Find a bounded, near-current, nonsingular joint solution and verify by FK."""
    current = np.asarray(current_joints, dtype=float)
    position = np.asarray(target_position, dtype=float)
    if current.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(current)):
        return None, 'current joint state is not six finite angles in JOINT_NAMES order'
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        return None, 'target position is not three finite base_link coordinates'
    if target_rotation.shape != (3, 3) or not np.all(np.isfinite(target_rotation)):
        return None, 'target orientation is not a finite 3x3 rotation matrix'

    margin = 0.03
    lower = UR7E_JOINT_LOWER_LIMITS + margin
    upper = UR7E_JOINT_UPPER_LIMITS - margin
    seeds = [current]
    for offsets in (
        (0.0, 0.8, -0.8, 0.0, 0.0, 0.0),
        (0.0, -0.8, 0.8, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.8, -0.8),
        (0.0, 0.0, 0.0, 0.0, -0.8, 0.8),
    ):
        seed = np.clip(current + np.asarray(offsets), lower, upper)
        if all(float(np.linalg.norm(seed - existing)) > 1e-3 for existing in seeds):
            seeds.append(seed)

    # Also start from shoulder-pan branches facing the package. These are only
    # optimizer initial guesses; every final joint vector is still solved and FK checked.
    target_bearing = math.atan2(position[1], position[0])
    bearing_seeds = []
    for turns in range(-2, 3):
        shoulder_pan = target_bearing + 2.0 * math.pi * turns
        if lower[0] <= shoulder_pan <= upper[0]:
            for wrist_turn in (-math.pi, 0.0, math.pi):
                seed = current.copy()
                seed[0] = shoulder_pan
                seed[5] = np.clip(
                    current[5] + wrist_turn, lower[5], upper[5]
                )
                if all(
                    float(np.linalg.norm(seed - existing)) > 1e-3
                    for existing in (*seeds, *bearing_seeds)
                ):
                    bearing_seeds.append(seed)
    bearing_seeds.sort(key=lambda seed: abs(float(seed[0] - current[0])))
    seeds.extend(bearing_seeds)

    solutions = []
    converged = 0
    largest_minimum_singular_value = 0.0
    for seed_index, seed in enumerate(seeds):
        joints = np.clip(seed.copy(), lower, upper)
        damping = 0.02
        for _ in range(160):
            weighted_error, position_error, orientation_error = _ik_residual(
                joints, position, target_rotation
            )
            if (
                float(np.linalg.norm(position_error)) <= IK_POSITION_TOLERANCE
                and float(np.linalg.norm(orientation_error)) <= IK_ORIENTATION_TOLERANCE
            ):
                transform = ur7e_tcp_transform(joints.tolist())
                singular_values = np.linalg.svd(
                    _ik_jacobian(joints, transform), compute_uv=False
                )
                minimum_singular_value = float(singular_values[-1])
                largest_minimum_singular_value = max(
                    largest_minimum_singular_value, minimum_singular_value
                )
                converged += 1
                if minimum_singular_value >= IK_MIN_SINGULAR_VALUE:
                    score = (
                        float(np.linalg.norm(joints - current))
                        + 0.02 / minimum_singular_value
                    )
                    solutions.append((score, joints.copy(), minimum_singular_value))
                    if (
                        seed_index == 0
                        and minimum_singular_value >= IK_PREFERRED_SINGULAR_VALUE
                    ):
                        break
                break

            transform = ur7e_tcp_transform(joints.tolist())
            jacobian = _ik_jacobian(joints, transform)
            try:
                update = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + damping * damping * np.eye(6),
                    weighted_error,
                )
            except np.linalg.LinAlgError:
                break

            update_norm = float(np.linalg.norm(update))
            if update_norm > 0.25:
                update *= 0.25 / update_norm

            previous_norm = float(np.linalg.norm(weighted_error))
            improved = False
            for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate = np.clip(joints + fraction * update, lower, upper)
                candidate_error, _, _ = _ik_residual(
                    candidate, position, target_rotation
                )
                if float(np.linalg.norm(candidate_error)) < previous_norm:
                    joints = candidate
                    damping = max(0.01, damping * 0.8)
                    improved = True
                    break
            if not improved:
                damping = min(0.5, damping * 2.0)
                if damping >= 0.5:
                    break

    if not solutions:
        if converged:
            return None, (
                f'{converged} converged candidate(s) rejected by minimum '
                f'singular value {IK_MIN_SINGULAR_VALUE:.3f}; best was '
                f'{largest_minimum_singular_value:.4f}'
            )
        return None, f'no solution converged from {len(seeds)} bounded IK seeds'

    _, solution, minimum_singular_value = min(solutions, key=lambda item: item[0])
    transform = ur7e_tcp_transform(solution.tolist())
    fk_position_error = float(np.linalg.norm(transform[:3, 3] - position))
    fk_orientation_error = float(np.linalg.norm(
        rotation_vector(target_rotation @ transform[:3, :3].T)
    ))
    within_limits = bool(np.all(solution >= lower) and np.all(solution <= upper))
    if (
        not within_limits
        or fk_position_error > IK_POSITION_TOLERANCE
        or fk_orientation_error > IK_ORIENTATION_TOLERANCE
    ):
        return None, (
            'forward-kinematics validation failed: '
            f'position error={fk_position_error:.6f} m, '
            f'orientation error={fk_orientation_error:.6f} rad, '
            f'joint limits respected={within_limits}'
        )
    return solution.tolist(), (
        f'FK verified (position error={fk_position_error:.6f} m, '
        f'orientation error={fk_orientation_error:.6f} rad); '
        f'minimum weighted Jacobian singular value={minimum_singular_value:.4f}'
    )


##################### CLASS DEFINITIONS #######################

class Task3BManipulation(Node):
    '''
    Description:    ROS2 Node for sequencing the complete Task 3B pick-and-sort pipeline:
                    unfolds arm, targets conveyor ores, commands Cartesian twist velocities
                    with active vertical orientation maintenance, operates the electromagnet,
                    verifies contact/grasp/lift/release forces, prioritizes the
                    eBot package at handover, then sorts any remaining ores.
    '''

    def __init__(self):
        super().__init__(
            'task3b_manipulation',
            parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)],
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)

        self.create_subscription(PoseStamped, '/tcp_pose_raw', self.tcp_pose_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_states_cb, 10)
        self.create_subscription(Int32, '/arm_status', self.arm_status_cb, 10)
        self.create_subscription(
            DiagnosticArray, '/arm_status_detail', self.arm_status_detail_cb, 10
        )
        self.create_subscription(Float32, '/ur7e/end_effector/force_magnitude', self.force_cb, 10)
        self.declare_parameter('crusher_y_offset', 0.0)
        self.switch_cli = self.create_client(SwitchController, '/ur_arm_controller/switch_controller')
        self.magnet_cli = self.create_client(SetBool, '/magnet')
        self.package_removed_pub = self.create_publisher(Bool, '/task3b/package_removed', 10)
        delivery_complete_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.package_delivery_complete_pub = self.create_publisher(
            Bool, '/task3b/package_delivery_complete', delivery_complete_qos
        )
        self.package_removed_announced = False
        self.package_delivery_complete_announced = False
        self.declare_parameter('crusher_x', CRUSHER_HOPPER_TARGET[0])
        self.declare_parameter('crusher_y', CRUSHER_HOPPER_TARGET[1])
        self.declare_parameter('crusher_z', CRUSHER_HOPPER_TARGET[2])
        self.declare_parameter('handover_x', 0.0)
        self.declare_parameter('handover_y', 1.9)
        self.declare_parameter('handover_radius', HANDOVER_POSITION_TOLERANCE_M)
        self.create_subscription(Odometry, '/odom', self.ebot_odom_cb, 10)
        self.ebot_xy: Optional[Tuple[float, float]] = None
        self.ebot_yaw: Optional[float] = None
        self.ebot_linear_speed: Optional[float] = None
        self.ebot_angular_speed: Optional[float] = None
        self.ebot_odom_received_at: Optional[float] = None
        self.handover_settle_started_at: Optional[float] = None
        self.package_delivery = False
        self.package_delivery_complete = False
        self.package_route_z: Optional[float] = None
        self.package_alignment_target_joints: Optional[List[float]] = None
        self.package_alignment_start: Optional[float] = None
        self.package_alignment_phase: Optional[str] = None
        self.package_alignment_destination: Optional[Tuple[float, float, float]] = None
        self.package_alignment_next_state = 26
        self.package_approach_target_position: Optional[Tuple[float, float, float]] = None
        self.package_approach_target_rotation: Optional[np.ndarray] = None
        self.package_ik_future: Optional[
            Future[Tuple[Optional[List[float]], str]]
        ] = None
        self.ik_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='task3b_package_ik'
        )
        self.last_missing_crusher_log = 0.0
        self.declare_parameter('package_alignment_timeout', 30.0)

        self.tcp_pose: Optional[PoseStamped] = None
        self.tcp_pose_received_at: Optional[float] = None
        self.joint_angles: Optional[List[float]] = None
        self.joint_states_received_at: Optional[float] = None
        self.arm_status: Optional[int] = None
        self.arm_status_received_at: Optional[float] = None
        self.arm_status_detail: Optional[str] = None
        self.force_magnitude = 0.0
        self.force_received_at: Optional[float] = None
        self.last_twist_linear = (0.0, 0.0, 0.0)
        self.last_twist_angular = (0.0, 0.0, 0.0)
        self.last_twist_time = self.now_sec()
        self.switch_future = None
        self.controller_list_cli = self.create_client(
            ListControllers, '/ur_arm_controller/list_controllers'
        )
        self.controller_list_future = None
        self.controller_states: Dict[str, str] = {}
        self.magnet_future = None
        # Recovery state serializes magnet-off requests; never discard an in-flight future.
        self.recovery_next_state = 4
        self.recovery_fault_reason: Optional[str] = None

        self.fsm_state = 0
        self.sorted_ores_completed: List[str] = []
        self.current_ore_name: Optional[str] = None
        self.current_ore_type: Optional[str] = None
        self.current_ore_pos: Optional[Tuple[float, float, float]] = None
        self.hover_pos: Optional[Tuple[float, float, float]] = None
        self.target_container_pos: Optional[Tuple[float, float, float]] = None
        self.pickup_surface_target: Optional[Tuple[float, float, float]] = None
        self.contact_start_z: Optional[float] = None
        self.state_timer_start: Optional[float] = None
        self.contact_sustain_start: Optional[float] = None
        self.grasp_attempts = 0
        self.transit_loss_start: Optional[float] = None
        self.protective_stop_latched = False
        self.fault_logged = False

        self.timer = self.create_timer(CONTROL_PERIOD, self.control_loop)
        self.get_logger().info('Task 3B Manipulation Node initialized.')

    def announce_package_removed(self):
        """Signal the eBot planner after the package is confirmed off its deck.

        Call this only from the package-pickup state after force/height checks confirm
        the package is held clear of the deck; merely seeing the package is insufficient.
        """
        if not self.package_removed_announced:
            msg = Bool()
            msg.data = True
            self.package_removed_pub.publish(msg)
            self.package_removed_announced = True
            self.get_logger().info('Handover signal published: package is off the eBot deck.')

    def monitor_package_retention(self, now: float, stage: str) -> bool:
        if not self.package_removed_announced:
            return True
        if self.force_magnitude >= 0.15:
            self.transit_loss_start = None
            return True
        if self.transit_loss_start is None:
            self.transit_loss_start = now
            return True
        if now - self.transit_loss_start <= 0.80:
            return True

        reason = (
            f'Package lost during {stage} '
            f'(force={self.force_magnitude:.2f} N); stopping delivery.'
        )
        self.publish_zero_twist()
        self.publish_zero_joint()
        self.recovery_next_state = 99
        self.recovery_fault_reason = reason
        self.fsm_state = 36
        self.get_logger().error(reason)
        return False

    def ebot_odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        quaternion = (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
        linear_speed = math.hypot(
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
        )
        angular_speed = abs(msg.twist.twist.angular.z)
        if (
            not all(math.isfinite(value) for value in (x, y, *quaternion))
            or not math.isfinite(quaternion_norm)
            or quaternion_norm < 1e-9
            or not math.isfinite(linear_speed)
            or not math.isfinite(angular_speed)
        ):
            self.get_logger().warning(
                'Ignoring invalid eBot pose or velocity feedback.',
                throttle_duration_sec=2.0,
            )
            return
        self.ebot_xy = (x, y)
        qx, qy, qz, qw = (
            value / quaternion_norm for value in quaternion
        )
        self.ebot_yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        self.ebot_linear_speed = linear_speed
        self.ebot_angular_speed = angular_speed
        self.ebot_odom_received_at = self.now_sec()

    def package_handover_pose(
        self, now: float
    ) -> Optional[Tuple[float, float, float]]:
        if (
            self.ebot_xy is None
            or self.ebot_yaw is None
            or self.ebot_linear_speed is None
            or self.ebot_angular_speed is None
            or self.ebot_odom_received_at is None
            or not 0.0 <= now - self.ebot_odom_received_at <= ODOMETRY_TIMEOUT_SEC
        ):
            self.handover_settle_started_at = None
            self.get_logger().info(
                f'Waiting for eBot odometry... received_at={self.ebot_odom_received_at}, now={now}',
                throttle_duration_sec=2.0,
            )
            return None

        handover_x = float(self.get_parameter('handover_x').value)
        handover_y = float(self.get_parameter('handover_y').value)
        handover_radius = float(self.get_parameter('handover_radius').value)
        if (
            not all(math.isfinite(value) for value in (
                handover_x, handover_y, handover_radius
            ))
            or handover_radius <= 0.0
        ):
            self.handover_settle_started_at = None
            self.fault('Handover coordinates/radius must be finite with a positive radius.')
            return None

        position_error = math.hypot(
            self.ebot_xy[0] - handover_x,
            self.ebot_xy[1] - handover_y,
        )
        if position_error > min(
            handover_radius, HANDOVER_POSITION_TOLERANCE_M
        ):
            self.handover_settle_started_at = None
            self.get_logger().info(
                f'Waiting for eBot to reach handover pose: error={position_error:.3f}m > {min(handover_radius, HANDOVER_POSITION_TOLERANCE_M):.3f}m (ebot_xy={self.ebot_xy})',
                throttle_duration_sec=2.0,
            )
            return None

        yaw_error = min(
            abs(math.atan2(math.sin(self.ebot_yaw), math.cos(self.ebot_yaw))),
            abs(math.atan2(
                math.sin(self.ebot_yaw - math.pi),
                math.cos(self.ebot_yaw - math.pi),
            )),
        )
        if (
            yaw_error > HANDOVER_YAW_TOLERANCE_RAD
            or self.ebot_linear_speed > HANDOVER_MAX_LINEAR_SPEED_MPS
            or self.ebot_angular_speed > HANDOVER_MAX_ANGULAR_SPEED_RPS
        ):
            self.handover_settle_started_at = None
            self.get_logger().info(
                f'Waiting for eBot to stabilize at handover: yaw_error={yaw_error:.3f}rad (tol={HANDOVER_YAW_TOLERANCE_RAD:.3f}), '
                f'lin_spd={self.ebot_linear_speed:.3f}m/s (tol={HANDOVER_MAX_LINEAR_SPEED_MPS:.3f}), '
                f'ang_spd={self.ebot_angular_speed:.3f}rad/s (tol={HANDOVER_MAX_ANGULAR_SPEED_RPS:.3f})',
                throttle_duration_sec=2.0,
            )
            return None

        if self.handover_settle_started_at is None:
            self.handover_settle_started_at = now
            return None
        if now - self.handover_settle_started_at < HANDOVER_SETTLE_DURATION_SEC:
            return None

        try:
            if not self.tf_buffer.can_transform(
                BASE_FRAME, 'ore_package', rclpy.time.Time()
            ):
                self.get_logger().info(
                    'eBot docked and settled, waiting for ore_package TF...',
                    throttle_duration_sec=2.0,
                )
                return None
            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME, 'ore_package', rclpy.time.Time()
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ExtrapolationException,
            tf2_ros.ConnectivityException,
        ) as exc:
            self.get_logger().info(
                f'ore_package TF lookup exception: {exc}',
                throttle_duration_sec=2.0,
            )
            return None

        stamp = transform.header.stamp
        transform_time = stamp.sec + stamp.nanosec * 1e-9
        if abs(now - transform_time) > PACKAGE_TF_TIMEOUT_SEC:
            self.get_logger().info(
                f'ore_package TF stale or clock skew: |now - tf_time| = {abs(now - transform_time):.3f}s > {PACKAGE_TF_TIMEOUT_SEC}s',
                throttle_duration_sec=2.0,
            )
            return None

        position = (
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        )
        if not all(math.isfinite(value) for value in position):
            self.get_logger().warning(
                'Ignoring ore_package TF with non-finite coordinates.',
                throttle_duration_sec=2.0,
            )
            return None
        return position

    def crusher_target(self) -> Optional[Tuple[float, float, float]]:
        y_offset = float(self.get_parameter('crusher_y_offset').value)
        x = float(self.get_parameter('crusher_x').value)
        y = float(self.get_parameter('crusher_y').value)
        z = float(self.get_parameter('crusher_z').value)
        if not all(math.isfinite(value) for value in (x, y, z, y_offset)):
            return None
        return x, y + y_offset, z

    def tcp_pose_cb(self, msg: PoseStamped):
        position = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        )
        orientation = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        quaternion_norm = math.sqrt(sum(value * value for value in orientation))
        if (
            msg.header.frame_id != BASE_FRAME
            or not all(math.isfinite(value) for value in position + orientation)
            or not math.isfinite(quaternion_norm)
            or quaternion_norm < 1e-9
        ):
            self.get_logger().warning(
                f'Ignoring invalid TCP feedback; expected a finite pose in {BASE_FRAME}.',
                throttle_duration_sec=2.0,
            )
            return
        msg.pose.orientation.x /= quaternion_norm
        msg.pose.orientation.y /= quaternion_norm
        msg.pose.orientation.z /= quaternion_norm
        msg.pose.orientation.w /= quaternion_norm
        self.tcp_pose = msg
        self.tcp_pose_received_at = self.now_sec()

    def arm_status_cb(self, msg: Int32):
        self.arm_status = msg.data
        self.arm_status_received_at = self.now_sec()

    def arm_status_detail_cb(self, msg: DiagnosticArray):
        for status in msg.status:
            if status.name != 'ur_arm_controller: safety':
                continue
            values = {item.key: item.value for item in status.values}
            reason = values.get('latched_reason', status.message)
            self.arm_status_detail = f'{status.message}: {reason}'
            if self.arm_status is not None and self.arm_status >= 20:
                self.get_logger().error(
                    f'Arm controller safety detail: {self.arm_status_detail}',
                    throttle_duration_sec=1.0,
                )
            return

    def force_cb(self, msg: Float32):
        if not math.isfinite(msg.data) or msg.data < 0.0:
            self.get_logger().warning(
                'Ignoring invalid end-effector force feedback.',
                throttle_duration_sec=2.0,
            )
            return
        self.force_magnitude = msg.data
        self.force_received_at = self.now_sec()

    def joint_states_cb(self, msg: JointState):
        name_to_position = dict(zip(msg.name, msg.position))
        if all(name in name_to_position for name in JOINT_NAMES):
            joint_angles = [name_to_position[name] for name in JOINT_NAMES]
            if not all(math.isfinite(value) for value in joint_angles):
                self.get_logger().warning(
                    'Ignoring non-finite arm joint feedback.',
                    throttle_duration_sec=2.0,
                )
                return
            self.joint_angles = joint_angles
            self.joint_states_received_at = self.now_sec()

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def publish_zero_twist(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = BASE_FRAME
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
        self.twist_pub.publish(msg)
        self.last_twist_linear = (0.0, 0.0, 0.0)
        self.last_twist_angular = (0.0, 0.0, 0.0)
        self.last_twist_time = self.now_sec()

    def publish_zero_joint(self):
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = JOINT_NAMES
        msg.velocities = [0.0 for _ in JOINT_NAMES]
        self.joint_pub.publish(msg)

    def compute_orientation_correction(self) -> Tuple[float, float, float]:
        '''
        Description:    Calculates angular velocity commands (wx, wy, wz) to actively drive
                        the tool pointing vector to [0, 0, -1] in base_link.
        '''
        if self.tcp_pose is None:
            return 0.0, 0.0, 0.0
        q = [
            self.tcp_pose.pose.orientation.x,
            self.tcp_pose.pose.orientation.y,
            self.tcp_pose.pose.orientation.z,
            self.tcp_pose.pose.orientation.w,
        ]
        r_curr = quat_to_rot_matrix(q)
        err = compute_tool_pointing_error(r_curr, target_axis=[0.0, 0.0, -1.0])
        kp_ang = 1.6
        wx = max(-0.35, min(0.35, kp_ang * err[0]))
        wy = max(-0.35, min(0.35, kp_ang * err[1]))
        wz = max(-0.35, min(0.35, kp_ang * err[2]))
        return wx, wy, wz

    def publish_twist(self, vx: float, vy: float, vz: float, apply_orientation: bool = True):
        '''
        Description:    Publishes linear velocity commands while applying closed-loop
                        angular velocity corrections to maintain vertical tool alignment.
        '''
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = BASE_FRAME
        desired_linear = (vx, vy, vz)
        desired_speed = math.sqrt(sum(component * component for component in desired_linear))
        if desired_speed > MAX_LINEAR_SPEED:
            scale = MAX_LINEAR_SPEED / desired_speed
            desired_linear = tuple(component * scale for component in desired_linear)
        if apply_orientation:
            desired_angular = self.compute_orientation_correction()
        else:
            desired_angular = (0.0, 0.0, 0.0)

        now = self.now_sec()
        dt = max(0.0, min(now - self.last_twist_time, 2.0 * CONTROL_PERIOD))
        limited_linear = self.limit_velocity_change(
            desired_linear, self.last_twist_linear, MAX_LINEAR_ACCELERATION, dt
        )
        limited_angular = self.limit_velocity_change(
            desired_angular, self.last_twist_angular, MAX_ANGULAR_ACCELERATION, dt
        )
        limited_speed = math.sqrt(sum(component * component for component in limited_linear))
        if limited_speed > MAX_LINEAR_SPEED:
            scale = MAX_LINEAR_SPEED / limited_speed
            limited_linear = tuple(component * scale for component in limited_linear)
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = limited_linear
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = limited_angular
        self.twist_pub.publish(msg)
        self.last_twist_linear = limited_linear
        self.last_twist_angular = limited_angular
        self.last_twist_time = now

    @staticmethod
    def limit_velocity_change(
        desired: Tuple[float, float, float],
        previous: Tuple[float, float, float],
        max_acceleration: float,
        dt: float,
    ) -> Tuple[float, float, float]:
        delta = tuple(desired[i] - previous[i] for i in range(3))
        delta_norm = math.sqrt(sum(component * component for component in delta))
        max_delta = max_acceleration * dt
        if delta_norm <= max_delta or delta_norm == 0.0:
            return desired
        scale = max_delta / delta_norm
        return tuple(previous[i] + scale * delta[i] for i in range(3))

    def request_controller_switch(self, activate: str, deactivate: str) -> bool:
        if not self.switch_cli.service_is_ready():
            return False
        req = SwitchController.Request()
        req.activate_controllers = [activate]
        req.deactivate_controllers = [deactivate]
        req.strictness = SwitchController.Request.STRICT
        req.timeout.sec = 2
        req.timeout.nanosec = 0
        self.switch_future = self.switch_cli.call_async(req)
        return True

    def poll_controller_states(self) -> Optional[Dict[str, str]]:
        if self.controller_list_future is None:
            if not self.controller_list_cli.service_is_ready():
                return None
            self.controller_list_future = self.controller_list_cli.call_async(
                ListControllers.Request()
            )
            return None
        if not self.controller_list_future.done():
            return None

        future = self.controller_list_future
        self.controller_list_future = None
        response = future.result()
        if response is None:
            raise RuntimeError('Controller list service returned no response.')
        self.controller_states = {
            controller.name: controller.state for controller in response.controller
        }
        return self.controller_states

    def controller_state_summary(self) -> str:
        if not self.controller_states:
            return 'unavailable'
        return ', '.join(
            f'{name}={state}' for name, state in sorted(self.controller_states.items())
        )

    def controller_mode_matches(self, active: str, inactive: str) -> bool:
        return (
            self.controller_states.get(active) == 'active'
            and self.controller_states.get(inactive) == 'inactive'
        )

    def set_magnet(self, enable: bool) -> bool:
        if not self.magnet_cli.service_is_ready():
            return False
        req = SetBool.Request()
        req.data = enable
        self.magnet_future = self.magnet_cli.call_async(req)
        return True

    def drive_tcp_towards(self, target: Tuple[float, float, float], tol: float = POSITION_TOLERANCE, max_speed: float = MAX_LINEAR_SPEED) -> bool:
        if self.tcp_pose is None:
            return False
        current = self.tcp_pose.pose.position
        ex, ey, ez = target[0] - current.x, target[1] - current.y, target[2] - current.z
        distance = math.sqrt(ex * ex + ey * ey + ez * ez)
        if distance <= tol:
            self.publish_zero_twist()
            return True

        gain = 2.5
        vx, vy, vz = gain * ex, gain * ey, gain * ez
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        speed_limit = min(max_speed, math.sqrt(2.0 * MAX_LINEAR_ACCELERATION * distance))
        if speed > speed_limit:
            scale = speed_limit / speed
            vx *= scale
            vy *= scale
            vz *= scale
        self.publish_twist(vx, vy, vz, apply_orientation=True)
        return False

    def fault(self, reason: str):
        self.publish_zero_twist()
        self.publish_zero_joint()
        self.fsm_state = 99
        if not self.fault_logged:
            self.fault_logged = True
            self.get_logger().error(f'TASK 3B MANIPULATION FAULT: {reason}')

    def control_loop(self):
        now = self.now_sec()
        feedback = (
            ('TCP pose', self.tcp_pose, self.tcp_pose_received_at),
            ('joint states', self.joint_angles, self.joint_states_received_at),
            ('arm safety status', self.arm_status, self.arm_status_received_at),
            ('end-effector force', self.force_magnitude, self.force_received_at),
        )
        missing_feedback = [
            name for name, value, received_at in feedback
            if value is None or received_at is None
        ]
        if missing_feedback:
            if self.fsm_state == 0:
                return
            self.fault(f'Missing required feedback: {", ".join(missing_feedback)}.')
            return
        stale_feedback = [
            name for name, _value, received_at in feedback
            if now < received_at or now - received_at > FEEDBACK_TIMEOUT_SEC
        ]
        if stale_feedback:
            if self.fsm_state == 0:
                return
            self.fault(f'Stale required feedback: {", ".join(stale_feedback)}.')
            return

        if self.arm_status >= 20:
            self.publish_zero_twist()
            self.publish_zero_joint()
            if not self.protective_stop_latched:
                self.protective_stop_latched = True
                detail = self.arm_status_detail or 'no /arm_status_detail message received'
                self.get_logger().error(
                    f'Protective stop active status={self.arm_status}; controller latched. '
                    f'Detail: {detail}'
                )
            self.fsm_state = 99
            return

        if (
            self.fsm_state == 5
            and self.current_ore_name != 'ore_package'
            and not self.package_removed_announced
            and not self.package_delivery_complete
        ):
            package_position = self.package_handover_pose(now)
            if self.fsm_state == 99:
                return
            if package_position is not None:
                self.publish_zero_twist()
                self.get_logger().info(
                    'Package appeared before ore grasp; safely cancelling the '
                    'unstarted pickup and prioritizing the package.'
                )
                self.current_ore_name = None
                self.current_ore_type = None
                self.current_ore_pos = None
                self.hover_pos = None
                self.pickup_surface_target = None
                self.package_delivery = False
                self.recovery_next_state = 23
                self.recovery_fault_reason = None
                self.fsm_state = 36
                return

        # State 0: Wait for controller manager service
        if self.fsm_state == 0:
            if not self.switch_cli.service_is_ready():
                return
            self.get_logger().info('SwitchController ready.')
            self.fsm_state = 1
            return

        # State 1: Activate delta_joint_controller for safe initial unfolding
        if self.fsm_state == 1:
            if self.switch_future is None:
                self.request_controller_switch(JOINT_CONTROLLER, TWIST_CONTROLLER)
                return
            if not self.switch_future.done():
                return
            try:
                result = self.switch_future.result()
                if result is not None and result.ok:
                    self.get_logger().info('Joint controller activated. Unfolding arm...')
                    self.switch_future = None
                    self.fsm_state = 2
                else:
                    self.fault('Failed to activate joint controller.')
            except Exception as exc:
                self.fault(f'Joint controller switch exception: {exc}')
            return

        # State 2: Unfold arm into ready stance
        if self.fsm_state == 2:
            if self.joint_angles is None:
                return
            velocities = []
            max_error = 0.0
            for i in range(len(JOINT_NAMES)):
                error = TARGET_UNFOLD_JOINTS[i] - self.joint_angles[i]
                max_error = max(max_error, abs(error))
                velocity = max(-JOINT_UNFOLD_SPEED, min(JOINT_UNFOLD_SPEED, 0.8 * error))
                velocities.append(velocity)
            if max_error <= JOINT_TOLERANCE:
                self.publish_zero_joint()
                self.get_logger().info('Arm safely unfolded into ready stance.')
                self.fsm_state = 3
                return
            msg = JointJog()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names = JOINT_NAMES
            msg.velocities = velocities
            self.joint_pub.publish(msg)
            return

        # State 3: Switch to delta_twist_controller for Cartesian tracking
        if self.fsm_state == 3:
            self.publish_zero_joint()
            if self.switch_future is None:
                self.get_logger().info('Switching to Cartesian Twist controller...')
                self.request_controller_switch(TWIST_CONTROLLER, JOINT_CONTROLLER)
                return
            if not self.switch_future.done():
                return
            try:
                result = self.switch_future.result()
                if result is not None and result.ok:
                    self.get_logger().info('Twist controller activated. Ready for ore sorting.')
                    self.switch_future = None
                    self.fsm_state = 4
                else:
                    self.fault('Failed to activate twist controller.')
            except Exception as exc:
                self.fault(f'Twist controller switch exception: {exc}')
            return

        # State 4: Prioritize a ready eBot package, otherwise select conveyor ore.
        if self.fsm_state == 4:
            self.publish_zero_twist()
            if self.package_delivery_complete and len(self.sorted_ores_completed) >= 4:
                self.get_logger().info(
                    'Package delivered and all four ores sorted; Task 3B manipulation complete.'
                )
                self.fsm_state = 99
                return

            package_position = None
            if (
                not self.package_removed_announced
                and not self.package_delivery_complete
            ):
                package_position = self.package_handover_pose(now)
                if self.fsm_state == 99:
                    return
                # If the package TF is present but stale, keep moving forward with
                # conveyor ore instead of stalling the FSM.
                if package_position is None and self.current_ore_name is None:
                    self.get_logger().info(
                        'Package TF absent or stale; proceeding with conveyor ore selection.',
                        throttle_duration_sec=2.0,
                    )
            if package_position is not None:
                self.get_logger().info(
                    'Ore package is visible at the eBot handover; prioritizing it '
                    'before selecting another conveyor ore.'
                )
                self.fsm_state = 23
                return

            candidates = [
                'azurite_ore_1', 'azurite_ore_2',
                'malachite_ore_1', 'malachite_ore_2',
                'vanadinite_ore_1', 'vanadinite_ore_2'
            ]
            selected_name = None
            selected_position = None
            for name in candidates:
                if name in self.sorted_ores_completed:
                    continue
                try:
                    if not self.tf_buffer.can_transform(BASE_FRAME, name, rclpy.time.Time()):
                        continue
                    transform = self.tf_buffer.lookup_transform(BASE_FRAME, name, rclpy.time.Time())
                    x = transform.transform.translation.x
                    y = transform.transform.translation.y
                    z = transform.transform.translation.z
                    if y < 0.0:  # Ore is in conveyor region
                        selected_name = name
                        selected_position = (x, y, z)
                        break
                except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException):
                    continue

            if selected_name is None:
                if len(self.sorted_ores_completed) >= 4:
                    if self.package_delivery_complete:
                        self.get_logger().info(
                            'All four ores sorted and package delivered; '
                            'Task 3B manipulation complete.'
                        )
                        self.fsm_state = 99
                        return
                    self.get_logger().info(
                        'All four ores sorted; waiting for the eBot package handover.'
                    )
                    self.fsm_state = 23
                return

            self.current_ore_name = selected_name
            self.current_ore_type = selected_name.split('_')[0]
            self.current_ore_pos = selected_position
            self.hover_pos = (selected_position[0], selected_position[1], selected_position[2] + 0.080)
            self.pickup_surface_target = (selected_position[0], selected_position[1], selected_position[2] + PICKUP_SURFACE_OFFSET)
            base_pos = CONTAINER_TARGETS[self.current_ore_type]
            count = sum(1 for o in self.sorted_ores_completed if o.startswith(self.current_ore_type))
            self.target_container_pos = (base_pos[0], base_pos[1] + (0.04 * count), base_pos[2])
            self.grasp_attempts = 0
            self.transit_loss_start = None
            self.contact_sustain_start = None
            self.contact_start_z = None

            self.get_logger().info(f'Targeting ore [{self.current_ore_name}] at {selected_position} -> container [{self.current_ore_type}]')
            self.get_logger().info(f'Pickup surface target: {self.pickup_surface_target}')
            self.fsm_state = 5
            return

        # State 5: Move to pre-grasp hover position above selected ore using latched coordinates
        if self.fsm_state == 5:
            reached = self.drive_tcp_towards(self.hover_pos, tol=POSITION_TOLERANCE, max_speed=MAX_LINEAR_SPEED)
            if reached:
                self.get_logger().info(f'Pre-grasp hover reached above [{self.current_ore_name}]. Activating magnet.')
                self.fsm_state = 6
            return

        # State 6: Energize electromagnet
        if self.fsm_state == 6:
            self.publish_zero_twist()
            if self.magnet_future is None:
                self.set_magnet(True)  # If service is not ready, retry on the next control tick.
                return
            if not self.magnet_future.done():
                return
            try:
                result = self.magnet_future.result()
                if result is not None and result.success:
                    self.get_logger().info('Electromagnet energized. Allowing transient to settle before descent.')
                    self.magnet_future = None
                    self.state_timer_start = now
                    self.fsm_state = 7
                else:
                    self.magnet_future = None
                    self.get_logger().warn('Magnet activation failed. Retrying...')
            except Exception as exc:
                self.magnet_future = None
                self.get_logger().error(f'Magnet activation exception: {exc}')
            return

        # State 7: Settle and descend towards pickup surface approach point
        if self.fsm_state == 7:
            if (now - self.state_timer_start) < MAGNET_SETTLE_TIME:
                self.publish_zero_twist()
                return
            if self.pickup_surface_target is None:
                self.fault('Pickup surface target is missing.')
                return

            reached = self.drive_tcp_towards(self.pickup_surface_target, tol=0.003, max_speed=DESCEND_SPEED)
            if reached:
                self.publish_zero_twist()
                self.get_logger().info(f'Pickup approach point reached. Force={self.force_magnitude:.2f} N. Initiating contact search.')
                self.contact_start_z = self.tcp_pose.pose.position.z
                self.contact_sustain_start = None
                self.fsm_state = 8
            return

        # State 8: Slow contact search & physical seating against ore top face
        if self.fsm_state == 8:
            current = self.tcp_pose.pose.position
            if self.contact_start_z is None:
                self.contact_start_z = current.z
            extra_descent = self.contact_start_z - current.z

            # Physical contact detection against compressed spring joint
            if self.force_magnitude >= FORCE_CONTACT_THRESHOLD:
                self.publish_zero_twist()
                if self.contact_sustain_start is None:
                    self.contact_sustain_start = now
                    self.get_logger().info(f'Pickup-zone contact force detected: {self.force_magnitude:.2f} N. Sustaining...')
                elif (now - self.contact_sustain_start) >= FORCE_CONTACT_SUSTAIN_TIME or self.force_magnitude >= FORCE_HOLD_TARGET:
                    self.state_timer_start = now
                    self.get_logger().info(f'Contact sustained at {self.force_magnitude:.2f} N. Proceeding to grasp verification.')
                    self.fsm_state = 10
                    return
                return
            else:
                # Hysteresis: only drop the sustain timer if force falls well
                # below the contact threshold, to avoid flickering right at the
                # boundary on rigid surfaces (package deck) or springy ore tops.
                if (
                    self.contact_sustain_start is not None
                    and self.force_magnitude < FORCE_CONTACT_THRESHOLD * 0.5
                ):
                    self.contact_sustain_start = None

            if extra_descent >= MAX_CONTACT_EXTRA_DESCENT:
                self.publish_zero_twist()
                self.grasp_attempts += 1
                self.get_logger().warn(
                    f'No solid ore contact detected after {extra_descent*1000:.1f} mm descent (attempt {self.grasp_attempts}). Retrying...'
                )
                self.current_ore_name = None
                self.package_delivery = False
                self.recovery_next_state = 35 if getattr(self, 'package_was_targeted', False) else 4
                self.recovery_fault_reason = (
                    'Exceeded maximum grasp contact retry attempts.' if self.grasp_attempts > 3 else None
                )
                self.fsm_state = 36  # Turn magnet off and await service result before recovery
                return

            self.publish_twist(0.0, 0.0, -SEATING_SPEED, apply_orientation=True)
            return

        # State 10: Dwelling and Grasp verification with retry recovery
        # Physical contact was already confirmed and sustained in State 8 (>= FORCE_CONTACT_THRESHOLD).
        # At zero velocity on a supported surface, resting compressive force naturally relaxes.
        # Grasp retention under load is robustly verified after lift in State 12.
        if self.fsm_state == 10:
            if (now - self.state_timer_start) < GRASP_SETTLE_TIME:
                self.publish_zero_twist()
                return
            force = self.force_magnitude
            if force < FORCE_CONTACT_THRESHOLD * 0.5:
                self.get_logger().warn(f'Grasp force dropped during settle ({force:.2f} N). Aborting grasp.')
                self.current_ore_name = None
                self.recovery_next_state = 4
                self.fsm_state = 36
                return
            self.get_logger().info(f'Grasp contact seated: force={force:.2f} N. Beginning lift.')
            self.fsm_state = 11
            return

        # State 11: Lift straight up in Z
        if self.fsm_state == 11:
            current = self.tcp_pose.pose.position
            lift_target = (current.x, current.y, LIFT_Z)
            reached = self.drive_tcp_towards(lift_target, tol=POSITION_TOLERANCE, max_speed=LIFT_SPEED)
            if reached:
                self.publish_zero_twist()
                self.state_timer_start = now
                self.get_logger().info(f'Ore [{self.current_ore_name}] lifted. Verifying retained force...')
                self.fsm_state = 12
            return

        # State 12: Verify lift retention — retreat to mid-corridor to re-acquire perception if dropped
        if self.fsm_state == 12:
            if (now - self.state_timer_start) < LIFT_VERIFY_TIME:
                self.publish_zero_twist()
                return
            force = self.force_magnitude
            self.get_logger().info(f'Lift verification force={force:.2f} N')
            min_verify_force = 0.10 if self.package_delivery else FORCE_GRASP_VERIFY
            if force < min_verify_force:
                self.grasp_attempts += 1
                self.get_logger().warn(
                    f'Ore lost during lift (force={force:.2f} N, attempt {self.grasp_attempts}). '
                    'Deactivating magnet and returning to mid-corridor to re-acquire target...'
                )
                self.current_ore_name = None
                self.pickup_surface_target = None
                self.package_delivery = False
                self.recovery_next_state = 21
                self.recovery_fault_reason = (
                    'Exceeded maximum grasp retry attempts.' if self.grasp_attempts > 3 else None
                )
                # Turn magnet off completely before returning to the corridor.
                self.fsm_state = 36
                return

            if self.package_delivery:
                self.announce_package_removed()
                self.package_route_z = LIFT_Z
                self.get_logger().info(
                    'Package retained at verified lift height; transiting to crusher via safe joint route (State 37).'
                )
                self.transit_loss_start = None
                self.fsm_state = 37
            else:
                self.get_logger().info(
                    f'Ore [{self.current_ore_name}] retained successfully. '
                    'Transiting to mid-corridor.'
                )
                self.fsm_state = 13
            return

        # State 13: Transit to safe mid-corridor waypoint at high Z with insignificant time
        if self.fsm_state == 13:
            if self.force_magnitude < 0.15:
                if self.transit_loss_start is None:
                    self.transit_loss_start = now
                elif (now - self.transit_loss_start) > 0.80:
                    self.get_logger().warn(f'Ore lost during corridor transit (force={self.force_magnitude:.2f} N). Returning to corridor...')
                    self.current_ore_name = None
                    self.recovery_next_state = 21
                    self.recovery_fault_reason = None
                    self.fsm_state = 36
                    return
                elif (now - self.transit_loss_start) > 0.15:
                    self.publish_zero_twist()
                    return
            else:
                self.transit_loss_start = None

            reached = self.drive_tcp_towards(
                WP_MID_CORRIDOR, tol=0.040, max_speed=MAX_LINEAR_SPEED
            )
            if reached:
                self.transit_loss_start = None
                self.fsm_state = 14
            return

        # State 14: Move in high-Z above target container
        if self.fsm_state == 14:
            if self.force_magnitude < 0.15:
                if self.transit_loss_start is None:
                    self.transit_loss_start = now
                elif (now - self.transit_loss_start) > 0.80:
                    self.get_logger().warn(f'Ore lost before container approach (force={self.force_magnitude:.2f} N). Returning to corridor...')
                    self.current_ore_name = None
                    self.recovery_next_state = 21
                    self.recovery_fault_reason = None
                    self.fsm_state = 36
                    return
                elif (now - self.transit_loss_start) > 0.15:
                    self.publish_zero_twist()
                    return
            else:
                self.transit_loss_start = None

            cx, cy, cz = self.target_container_pos
            reached = self.drive_tcp_towards(
                (cx, cy, cz),
                tol=POSITION_TOLERANCE,
                max_speed=CONTAINER_SPEED,
            )
            if reached:
                self.publish_zero_twist()
                self.get_logger().info(
                    f'Container safe release height reached at ({cx:.4f}, {cy:.4f}, {cz:.4f}). Directly de-energizing magnet.'
                )
                self.transit_loss_start = None
                self.fsm_state = 15
            return

        # State 22: Descend straight down in Z to container release height
        if self.fsm_state == 22:
            # A transient low force reading must not stall the release descent.
            # Keep the controlled vertical release motion moving; state 16
            # verifies that force falls after the magnet is switched off.
            cx, cy, cz = self.target_container_pos
            reached = self.drive_tcp_towards(
                (cx, cy, cz),
                tol=0.005,
                max_speed=CONTAINER_DESCENT_SPEED,
            )
            if reached:
                self.publish_zero_twist()
                self.get_logger().info(
                    f'Container release point reached at ({cx:.4f}, {cy:.4f}, {cz:.4f}). De-energizing magnet.'
                )
                self.transit_loss_start = None
                self.fsm_state = 15
            return

        # State 15: Deactivate magnet
        if self.fsm_state == 15:
            self.publish_zero_twist()
            if self.magnet_future is None:
                self.set_magnet(False)  # If service is not ready, retry on the next control tick.
                return
            if not self.magnet_future.done():
                return
            try:
                result = self.magnet_future.result()
                if result is not None and result.success:
                    self.get_logger().info('Magnet OFF. Verifying release.')
                    self.magnet_future = None
                    self.state_timer_start = now
                    self.fsm_state = 16
                else:
                    self.fault('Magnet OFF service failed.')
            except Exception as exc:
                self.fault(f'Magnet release exception: {exc}')
            return

        # State 16: Verify release
        if self.fsm_state == 16:
            self.publish_zero_twist()
            elapsed = now - self.state_timer_start
            force = self.force_magnitude
            if elapsed < RELEASE_SETTLE_TIME:
                return
            if force <= FORCE_RELEASED:
                self.get_logger().info(
                    f'Ore [{self.current_ore_name}] released successfully into [{self.current_ore_type}]. Force={force:.2f} N'
                )
                if not self.package_delivery:
                    self.sorted_ores_completed.append(self.current_ore_name)
                # If all work is complete, finish immediately without returning to corridor
                if (
                    len(self.sorted_ores_completed) >= 4
                    and (self.package_delivery_complete or self.package_delivery)
                ):
                    self.publish_zero_twist()
                    if self.package_delivery:
                        self.package_delivery = False
                        self.package_delivery_complete = True
                    self.get_logger().info(
                        'Package delivered and all four ores sorted; '
                        'Task 3B manipulation complete.'
                    )
                    self.fsm_state = 99
                    return
                self.state_timer_start = now
                self.fsm_state = 17
                return
            if elapsed > 1.50:
                if force <= 0.65:
                    self.get_logger().info(
                        f'Ore [{self.current_ore_name}] release accepted at timeout. Force={force:.2f} N'
                    )
                    if not self.package_delivery:
                        self.sorted_ores_completed.append(self.current_ore_name)
                    self.state_timer_start = now
                    self.fsm_state = 17
                    return
                self.fault(f'Ore release not verified. Force={force:.2f} N after {elapsed:.2f}s')
                return
            return

        # State 17: Ascend straight up in Z to high-Z before leaving container
        if self.fsm_state == 17:
            cx, cy, cz = self.target_container_pos
            reached = self.drive_tcp_towards(
                (cx, cy, LIFT_Z),
                tol=TRANSIT_TOLERANCE,
                max_speed=MAX_LINEAR_SPEED,
            )
            if reached:
                if self.package_delivery:
                    # Package drop at crusher: NO MID CORRIDOR FOR ORE PACKAGE!
                    self.publish_zero_twist()
                    self.get_logger().info(
                        'Package delivered to crusher; bypassing mid-corridor completely.'
                    )
                    self.package_delivery = False
                    self.package_delivery_complete = True
                    if not self.package_delivery_complete_announced:
                        msg = Bool()
                        msg.data = True
                        self.package_delivery_complete_pub.publish(msg)
                        self.package_delivery_complete_announced = True
                        self.get_logger().info('Package delivery complete announced immediately from crusher drop.')
                    self.current_ore_name = None
                    self.current_ore_type = None
                    self.current_ore_pos = None
                    self.hover_pos = None
                    self.pickup_surface_target = None
                    if len(self.sorted_ores_completed) >= 4:
                        self.get_logger().info('Package delivered and all four ores sorted; Task 3B manipulation complete.')
                        self.fsm_state = 99
                    else:
                        self.get_logger().info('Package delivered from crusher; switching to joint controller to reset arm pose safely.')
                        self.fsm_state = 35  # Transition through joint-space reset/unfold
                else:
                    # Regular ore dropped in container
                    if len(self.sorted_ores_completed) >= 4:
                        self.publish_zero_twist()
                        if self.package_delivery_complete:
                            self.get_logger().info(
                                'Package delivered and all four ores sorted; '
                                'Task 3B manipulation complete.'
                            )
                            self.fsm_state = 99
                        else:
                            self.get_logger().info(
                                'All four ores sorted; proceeding directly to eBot package pickup without corridor.'
                            )
                            self.fsm_state = 23
                    elif (
                        (
                            not self.package_removed_announced
                            and not self.package_delivery_complete
                            and not self.package_delivery
                            and self.package_handover_pose(now) is not None
                        )
                    ):
                        # Ore package is ready: bypass mid-corridor and go straight to package pickup!
                        self.publish_zero_twist()
                        self.get_logger().info(
                            'Ore package pickup requested; proceeding directly to package pickup without corridor.'
                        )
                        self.current_ore_name = None
                        self.current_ore_type = None
                        self.current_ore_pos = None
                        self.hover_pos = None
                        self.pickup_surface_target = None
                        self.fsm_state = 23
                    else:
                        # For regular ores: keep mid corridor with insignificant time
                        self.fsm_state = 21
            return

        # State 21: Return in high-Z to mid-corridor with insignificant time
        if self.fsm_state == 21:
            # Check package delivery first before waiting at mid-corridor
            if (
                not self.package_removed_announced
                and not self.package_delivery_complete
                and not self.package_delivery
                and self.package_handover_pose(now) is not None
            ):
                self.publish_zero_twist()
                self.get_logger().info('Ore package pickup requested; switching directly to package collection.')
                self.current_ore_name = None
                self.current_ore_type = None
                self.current_ore_pos = None
                self.hover_pos = None
                self.pickup_surface_target = None
                self.fsm_state = 23
                return

            reached = self.drive_tcp_towards(
                WP_MID_CORRIDOR, tol=0.040, max_speed=MAX_LINEAR_SPEED
            )
            if reached:
                if len(self.sorted_ores_completed) >= 4:
                    self.publish_zero_twist()
                    if self.package_delivery_complete:
                        self.get_logger().info(
                            'Package delivered and all four ores sorted; '
                            'Task 3B manipulation complete.'
                        )
                        self.fsm_state = 99
                    else:
                        self.get_logger().info(
                            'All four ores sorted; proceeding directly to eBot package handover.'
                        )
                        self.fsm_state = 23
                else:
                    self.current_ore_name = None
                    self.current_ore_type = None
                    self.current_ore_pos = None
                    self.hover_pos = None
                    self.pickup_surface_target = None
                    self.fsm_state = 4
            return

        # State 23: Wait for and initialize the prioritized eBot package pick/place flow.
        if self.fsm_state == 23:
            self.publish_zero_twist()

            package_position = self.package_handover_pose(now)
            if package_position is None:
                return

            px, py, pz = package_position
            self.target_container_pos = self.crusher_target()
            if self.target_container_pos is None:
                self.fault('Crusher coordinates are invalid for package delivery.')
                return

            self.current_ore_name = 'ore_package'
            self.current_ore_type = 'crusher'
            self.current_ore_pos = (px, py, pz)
            self.hover_pos = (px, py, pz + 0.080)
            self.pickup_surface_target = (
                px, py, pz + PICKUP_SURFACE_OFFSET
            )
            self.package_route_z = LIFT_Z
            self.package_delivery = True
            self.package_was_targeted = True
            self.grasp_attempts = 0
            self.contact_start_z = None
            self.contact_sustain_start = None
            self.transit_loss_start = None

            self.get_logger().info(
                f'eBot handover detected; using perception TF in {BASE_FRAME}: '
                f'package target={self.pickup_surface_target}, '
                f'crusher target={self.target_container_pos}.'
            )
            self.get_logger().info(
                'Package approach activated: lifting to clearance height (State 24).'
            )
            self.fsm_state = 24
            return

        # State 24: Reach the package route's clearance height before solving its IK pose.
        if self.fsm_state == 24:
            if self.package_route_z is None or self.current_ore_pos is None:
                self.fault('Package approach route is missing its target/clearance height.')
                return
            current = self.tcp_pose.pose.position
            target = (current.x, current.y, self.package_route_z)
            reached = self.drive_tcp_towards(
                target, tol=TRANSIT_TOLERANCE, max_speed=CONTAINER_SPEED
            )
            if reached:
                self.publish_zero_twist()
                self.get_logger().info(
                    'Package approach: clearance height reached; solving TCP pose IK.'
                )
                px, py, _ = self.current_ore_pos
                self.package_alignment_destination = (
                    px, py, self.package_route_z
                )
                self.package_alignment_next_state = 26
                self.package_alignment_start = None
                self.package_alignment_phase = None
                self.package_alignment_target_joints = None
                self.package_approach_target_position = None
                self.package_approach_target_rotation = None
                self.fsm_state = 40
            return

        # State 40: Solve and validate the package clearance pose, then switch to JointJog.
        if self.fsm_state == 40:
            if self.package_alignment_destination is None:
                self.fault('Package IK target or clearance height is missing.')
                return
            if not self.monitor_package_retention(now, 'joint-space transfer'):
                return

            if self.package_alignment_start is None:
                self.package_alignment_start = now
                self.package_alignment_phase = 'check_cartesian'

            timeout = float(
                self.get_parameter('package_alignment_timeout').value
            )
            if not math.isfinite(timeout) or timeout <= 0.0:
                self.fault('package_alignment_timeout must be finite and positive.')
                return
            if now - self.package_alignment_start > timeout:
                self.fault(
                    'Package IK/controller preparation timed out; '
                    f'controller states: {self.controller_state_summary()}.'
                )
                return

            if self.package_alignment_phase in ('check_cartesian', 'wait_for_ik'):
                self.publish_zero_twist()

            if self.joint_angles is None:
                return

            if self.package_alignment_phase == 'check_cartesian':
                try:
                    states = self.poll_controller_states()
                except Exception as exc:
                    self.fault(f'Could not query package controller state: {exc}')
                    return
                if states is None:
                    return
                if not self.controller_mode_matches(TWIST_CONTROLLER, JOINT_CONTROLLER):
                    self.fault(
                        'Package IK requires Cartesian mode before switching; '
                        f'controller states: {self.controller_state_summary()}.'
                    )
                    return
                if self.tcp_pose.header.frame_id != BASE_FRAME:
                    self.fault(
                        f'Package IK requires /tcp_pose_raw in {BASE_FRAME}; '
                        f'got {self.tcp_pose.header.frame_id!r}.'
                    )
                    return

                self.package_approach_target_position = (
                    self.package_alignment_destination
                )
                try:
                    quaternion = [
                        self.tcp_pose.pose.orientation.x,
                        self.tcp_pose.pose.orientation.y,
                        self.tcp_pose.pose.orientation.z,
                        self.tcp_pose.pose.orientation.w,
                    ]
                    self.package_approach_target_rotation = (
                        rotation_from_quaternion(quaternion)
                    )
                except ValueError as exc:
                    self.fault(
                        f'Package IK target orientation is invalid: {exc}; '
                        f'controller states: {self.controller_state_summary()}.'
                    )
                    return

                try:
                    self.package_ik_future = self.ik_executor.submit(
                        solve_ur7e_ik,
                        self.joint_angles.copy(),
                        self.package_approach_target_position,
                        self.package_approach_target_rotation.copy(),
                    )
                except RuntimeError as exc:
                    self.fault(
                        f'Could not start package IK worker: {exc}; '
                        f'controller states: {self.controller_state_summary()}.'
                    )
                    return

                self.package_alignment_phase = 'wait_for_ik'
                return

            if self.package_alignment_phase == 'wait_for_ik':
                if self.package_ik_future is None:
                    self.fault('Package IK worker future is missing.')
                    return
                if not self.package_ik_future.done():
                    return
                future = self.package_ik_future
                self.package_ik_future = None
                try:
                    solution, ik_result = future.result()
                except Exception as exc:
                    self.fault(
                        f'Package IK worker failed: {exc}; '
                        f'target_pose=({self.package_approach_target_position}, '
                        f'rotation={self.package_approach_target_rotation.tolist()}), '
                        f'current_joints={self.joint_angles}, '
                        f'controller_states={self.controller_state_summary()}.'
                    )
                    return
                if solution is None:
                    self.grasp_attempts += 1
                    if self.grasp_attempts > 5:
                        self.fault(
                            'Package IK failed after 5 retries: '
                            f'{ik_result}; target_pose=('
                            f'{self.package_approach_target_position}, '
                            f'rotation={self.package_approach_target_rotation.tolist()}), '
                            f'current_joints={self.joint_angles}, '
                            f'controller_states={self.controller_state_summary()}.'
                        )
                        return
                    self.get_logger().warn(
                        f'Package IK failed (attempt {self.grasp_attempts}/5): {ik_result}. '
                        'Waiting for dock settling and retrying with fresh perception TF...'
                    )
                    self.package_alignment_phase = None
                    self.package_alignment_start = None
                    self.package_ik_future = None
                    self.fsm_state = 23
                    return

                self.package_alignment_target_joints = solution
                self.get_logger().info(
                    'Package IK solution validated: '
                    f'target={self.package_approach_target_position}, '
                    f'joints={solution}; {ik_result}.'
                )
                self.package_alignment_phase = 'switch_to_joint'
                return

            if self.package_alignment_phase == 'switch_to_joint':
                if self.switch_future is None:
                    self.publish_zero_twist()
                    if not self.request_controller_switch(
                        JOINT_CONTROLLER, TWIST_CONTROLLER
                    ):
                        return
                    self.package_alignment_phase = 'wait_for_joint_switch'
                    return

            if self.package_alignment_phase == 'wait_for_joint_switch':
                if self.switch_future is None or not self.switch_future.done():
                    return
                try:
                    result = self.switch_future.result()
                except Exception as exc:
                    self.switch_future = None
                    self.fault(f'Package joint-controller switch exception: {exc}')
                    return
                self.switch_future = None
                if result is None or not result.ok:
                    self.fault(
                        'Failed to activate the package JointJog controller; '
                        f'controller states: {self.controller_state_summary()}.'
                    )
                    return
                self.package_alignment_phase = 'verify_joint_controller'
                return

            if self.package_alignment_phase == 'verify_joint_controller':
                try:
                    states = self.poll_controller_states()
                except Exception as exc:
                    self.fault(f'Could not verify package JointJog controller: {exc}')
                    return
                if states is None:
                    return
                if not self.controller_mode_matches(JOINT_CONTROLLER, TWIST_CONTROLLER):
                    self.fault(
                        'Package JointJog switch returned success but the expected '
                        f'mode is not active; controller states: '
                        f'{self.controller_state_summary()}.'
                    )
                    return
                self.get_logger().info(
                    'Package JointJog controller confirmed active; executing IK target.'
                )
                self.package_alignment_phase = 'move_to_ik_target'
                self.fsm_state = 41
                return

            self.fault(
                f'Unknown package IK phase {self.package_alignment_phase!r}; '
                f'controller states: {self.controller_state_summary()}.'
            )
            return

        # State 41: Execute the validated IK solution through the active JointJog mode.
        if self.fsm_state == 41:
            if not self.monitor_package_retention(now, 'joint-space transfer'):
                return
            target_joints = self.package_alignment_target_joints
            if target_joints is None:
                self.fault(
                    'Package joint target is missing; '
                    f'controller states: {self.controller_state_summary()}.'
                )
                return

            timeout = float(
                self.get_parameter('package_alignment_timeout').value
            )
            if (
                not math.isfinite(timeout)
                or timeout <= 0.0
                or self.package_alignment_start is None
                or now - self.package_alignment_start > timeout
            ):
                self.fault(
                    'Package joint-space IK motion timed out; '
                    f'controller states: {self.controller_state_summary()}.'
                )
                return

            if self.joint_angles is None:
                return
            if not self.controller_mode_matches(JOINT_CONTROLLER, TWIST_CONTROLLER):
                self.fault(
                    'Package joint motion stopped because JointJog is not the active '
                    f'controller mode; controller states: {self.controller_state_summary()}.'
                )
                return

            errors = [
                target_joints[i] - self.joint_angles[i]
                for i in range(len(JOINT_NAMES))
            ]
            max_error = max(abs(error) for error in errors)
            if max_error <= PACKAGE_IK_JOINT_TOLERANCE:
                self.publish_zero_joint()
                self.get_logger().info(
                    'Package IK joint target reached. '
                    f'Joint angles={self.joint_angles}'
                )
                self.package_alignment_phase = 'switch_to_twist'
                self.fsm_state = 42
                return

            velocities = [
                max(
                    -JOINT_UNFOLD_SPEED,
                    min(JOINT_UNFOLD_SPEED, 4.0 * error)
                )
                for error in errors
            ]
            msg = JointJog()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names = JOINT_NAMES
            msg.velocities = velocities
            self.joint_pub.publish(msg)
            return

        # State 42: Restore Cartesian mode, verify the achieved TCP pose, then descend.
        if self.fsm_state == 42:
            if not self.monitor_package_retention(now, 'joint-space transfer'):
                return
            timeout = float(
                self.get_parameter('package_alignment_timeout').value
            )
            if (
                not math.isfinite(timeout)
                or timeout <= 0.0
                or self.package_alignment_start is None
                or now - self.package_alignment_start > timeout
            ):
                self.fault(
                    'Package controller restore/TCP verification timed out; '
                    f'controller states: {self.controller_state_summary()}.'
                )
                return

            if self.package_alignment_phase == 'switch_to_twist':
                if self.switch_future is None:
                    if not self.request_controller_switch(
                        TWIST_CONTROLLER, JOINT_CONTROLLER
                    ):
                        return
                    self.package_alignment_phase = 'wait_for_twist_switch'
                    return
            if self.package_alignment_phase == 'wait_for_twist_switch':
                if self.switch_future is None or not self.switch_future.done():
                    return
                try:
                    result = self.switch_future.result()
                except Exception as exc:
                    self.switch_future = None
                    self.fault(f'Cartesian controller restore exception after package IK: {exc}')
                    return
                self.switch_future = None
                if result is None or not result.ok:
                    self.fault(
                        'Failed to restore Cartesian controller after package IK; '
                        f'controller states: {self.controller_state_summary()}.'
                    )
                    return
                self.package_alignment_phase = 'verify_twist_controller'
                return

            if self.package_alignment_phase == 'verify_twist_controller':
                try:
                    states = self.poll_controller_states()
                except Exception as exc:
                    self.fault(f'Could not verify Cartesian controller after package IK: {exc}')
                    return
                if states is None:
                    return
                if not self.controller_mode_matches(TWIST_CONTROLLER, JOINT_CONTROLLER):
                    self.fault(
                        'Cartesian controller switch returned success but the expected '
                        f'mode is not active; controller states: '
                        f'{self.controller_state_summary()}.'
                    )
                    return

                if (
                    self.tcp_pose is None
                    or self.tcp_pose.header.frame_id != BASE_FRAME
                    or self.joint_angles is None
                    or self.package_approach_target_position is None
                    or self.package_approach_target_rotation is None
                    or self.package_alignment_target_joints is None
                ):
                    self.fault(
                        'Package IK execution verification is missing TCP, joint, or target '
                        f'feedback; controller states: {self.controller_state_summary()}.'
                    )
                    return
                try:
                    actual_rotation = rotation_from_quaternion([
                        self.tcp_pose.pose.orientation.x,
                        self.tcp_pose.pose.orientation.y,
                        self.tcp_pose.pose.orientation.z,
                        self.tcp_pose.pose.orientation.w,
                    ])
                except ValueError as exc:
                    self.fault(f'Package TCP verification orientation is invalid: {exc}.')
                    return
                actual_position = np.array([
                    self.tcp_pose.pose.position.x,
                    self.tcp_pose.pose.position.y,
                    self.tcp_pose.pose.position.z,
                ])
                if (
                    not np.all(np.isfinite(actual_position))
                    or not all(math.isfinite(value) for value in self.joint_angles)
                ):
                    self.fault(
                        'Package IK execution feedback contains non-finite values; '
                        f'controller states: {self.controller_state_summary()}.'
                    )
                    return
                position_error = float(np.linalg.norm(
                    actual_position - np.asarray(self.package_approach_target_position)
                ))
                orientation_error = float(np.linalg.norm(rotation_vector(
                    self.package_approach_target_rotation @ actual_rotation.T
                )))
                joint_error = max(
                    abs(self.package_alignment_target_joints[index] - self.joint_angles[index])
                    for index in range(len(JOINT_NAMES))
                )
                if (
                    position_error > TRANSIT_TOLERANCE
                    or orientation_error > 0.03
                    or joint_error > JOINT_TOLERANCE
                ):
                    self.fault(
                        'Package IK execution did not reach its validated pose: '
                        f'position error={position_error:.4f} m, '
                        f'orientation error={orientation_error:.4f} rad, '
                        f'joint error={joint_error:.4f} rad; '
                        f'controller states={self.controller_state_summary()}.'
                    )
                    return

                self.get_logger().info(
                    'Package IK transfer verified from TCP and joint feedback '
                    f'(position error={position_error:.4f} m, '
                    f'orientation error={orientation_error:.4f} rad, '
                    f'joint error={joint_error:.4f} rad).'
                )
                next_state = self.package_alignment_next_state
                self.package_alignment_phase = None
                self.package_alignment_start = None
                self.package_alignment_destination = None
                self.package_alignment_next_state = 26
                self.fsm_state = next_state
                return

            self.fault(
                f'Unknown package controller/TCP verification phase '
                f'{self.package_alignment_phase!r}; '
                f'controller states: {self.controller_state_summary()}.'
            )
            return

        # State 26: Descend vertically to the package hover pose, then immediately energize magnet (State 6).
        if self.fsm_state == 26:
            if self.hover_pos is None:
                self.fault('Package hover target is missing.')
                return
            px, py, pz = self.hover_pos
            reached = self.drive_tcp_towards(
                (px, py, pz), tol=POSITION_TOLERANCE, max_speed=DESCEND_SPEED
            )
            if reached:
                self.publish_zero_twist()
                self.get_logger().info('Package hover reached; immediately energizing magnet.')
                self.fsm_state = 6
            return

        # States 37-39: Staged high-clearance route from the eBot to the crusher.
        if self.fsm_state == 37:
            if not self.monitor_package_retention(now, 'clearance lift'):
                return
            if self.target_container_pos is None:
                self.fault('Crusher target is missing for package delivery.')
                return
            if self.package_route_z is None:
                self.package_route_z = max(LIFT_Z + 0.100, self.target_container_pos[2] + 0.250)
            current = self.tcp_pose.pose.position
            reached = self.drive_tcp_towards(
                (current.x, current.y, self.package_route_z),
                tol=TRANSIT_TOLERANCE, max_speed=PACKAGE_TRANSFER_SPEED
            )
            if reached:
                self.publish_zero_twist()
                cx, cy, _ = self.target_container_pos
                approach_x = cx + math.copysign(
                    min(CRUSHER_APPROACH_X_OFFSET, abs(cx)), -cx
                )
                self.package_alignment_destination = (
                    approach_x, cy, self.package_route_z
                )
                self.package_alignment_next_state = 38
                self.package_alignment_start = None
                self.package_alignment_phase = None
                self.package_alignment_target_joints = list(CRUSHER_APPROACH_SEED_JOINTS)
                self.package_approach_target_position = None
                self.package_approach_target_rotation = None
                self.get_logger().info(
                    'Crusher route: clearance height reached; solving IK for the '
                )
                self.fsm_state = 40
            return

        if self.fsm_state == 38:
            if not self.monitor_package_retention(now, 'crusher alignment'):
                return
            if self.target_container_pos is None or self.package_route_z is None:
                self.fault('Crusher staged-route target is missing.')
                return
            cx, cy, _ = self.target_container_pos
            approach_x = cx + math.copysign(
                min(CRUSHER_APPROACH_X_OFFSET, abs(cx)), -cx
            )
            reached = self.drive_tcp_towards(
                (approach_x, cy, self.package_route_z),
                tol=TRANSIT_TOLERANCE, max_speed=PACKAGE_TRANSFER_SPEED
            )
            if reached:
                self.publish_zero_twist()
                self.get_logger().info('Crusher route: aligned above hopper; descending to release hover.')
                self.fsm_state = 39
            return

        if self.fsm_state == 39:
            if not self.monitor_package_retention(now, 'crusher release'):
                return
            if self.target_container_pos is None:
                self.fault('Crusher release target is missing.')
                return
            cx, cy, cz = self.target_container_pos
            reached = self.drive_tcp_towards(
                (cx, cy, cz), tol=TRANSIT_TOLERANCE, max_speed=CONTAINER_SPEED
            )
            if reached:
                self.publish_zero_twist()
                self.get_logger().info(
                    f'Crusher safe release height reached at ({cx:.4f}, {cy:.4f}, {cz:.4f}). Directly de-energizing magnet.'
                )
                self.transit_loss_start = None
                self.fsm_state = 15
            return
        # State 35: Switch to joint controller and reset arm to unfold stance after crusher delivery
        if self.fsm_state == 35:
            self.publish_zero_twist()
            if self.switch_future is None:
                self.request_controller_switch(JOINT_CONTROLLER, TWIST_CONTROLLER)
                return
            if not self.switch_future.done():
                return
            try:
                result = self.switch_future.result()
                if result is not None and result.ok:
                    self.switch_future = None
                    self.fsm_state = 351
                else:
                    self.fault('Failed to activate joint controller for post-crusher reset.')
            except Exception as exc:
                self.fault(f'Joint controller switch exception: {exc}')
            return

        if self.fsm_state == 351:
            if self.joint_angles is None:
                return
            velocities = []
            max_error = 0.0
            for i in range(len(JOINT_NAMES)):
                error = TARGET_UNFOLD_JOINTS[i] - self.joint_angles[i]
                max_error = max(max_error, abs(error))
                velocity = max(-JOINT_UNFOLD_SPEED, min(JOINT_UNFOLD_SPEED, 0.8 * error))
                velocities.append(velocity)
            if max_error <= JOINT_TOLERANCE:
                self.publish_zero_joint()
                self.get_logger().info('Arm reset to ready stance; switching back to twist controller.')
                self.fsm_state = 3  # Switch back to twist controller and resume State 4
                return
            msg = JointJog()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names = JOINT_NAMES
            msg.velocities = velocities
            self.joint_pub.publish(msg)
            return
        # State 36: Serialize magnet-off recovery requests.
        # The pending service future is retained until it completes; this prevents a
        # later grasp state from mistaking an old magnet-OFF result for magnet-ON.
        if self.fsm_state == 36:
            self.publish_zero_twist()
            if self.magnet_future is None:
                if not self.set_magnet(False):
                    return  # Service not ready yet; retry without moving the arm.
                return
            if not self.magnet_future.done():
                return
            try:
                result = self.magnet_future.result()
            except Exception as exc:
                self.magnet_future = None
                self.fault(f'Magnet-OFF recovery service exception: {exc}')
                return
            self.magnet_future = None
            if result is None or not result.success:
                self.fault('Magnet-OFF recovery service failed; arm movement halted.')
                return
            if self.recovery_fault_reason:
                reason = self.recovery_fault_reason
                self.recovery_fault_reason = None
                self.fault(reason)
                return
            self.get_logger().info('Magnet OFF confirmed; resuming recovery path.')
            self.fsm_state = self.recovery_next_state
            return

        # State 99: Latched terminal/fault state
        if self.fsm_state == 99:
            self.publish_zero_twist()
            return


##################### MAIN FUNCTION #######################

def main(args=None):
    rclpy.init(args=args)
    node = Task3BManipulation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            try:
                node.publish_zero_twist()
                node.publish_zero_joint()
            except Exception:
                pass
        node.ik_executor.shutdown(wait=True, cancel_futures=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
