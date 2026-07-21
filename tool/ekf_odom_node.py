#!/usr/bin/env python3
"""
Topics
------
  Predict:  /camera/camera/vio_image  → ekf_predict  (20Hz)  → /slam/odometry_fused
            /camera/camera/vio_100hz  → ekf_predict  (100Hz) → /slam/odometry_fused_100hz
  Update:   /robot/odom, /robot_odom  → ekf_update_pose6  (6-DOF, delta-referenced, huge-noise z)
            /wheel/odom_camera        → ekf_update_pose6  (6-DOF, delta-referenced)
            /lidar/odom_camera        → ekf_update_pose6  (6-DOF, delta-referenced)
            /qr/odom                  → ekf_update_pose6  (6-DOF, absolute)
            /rtk/odom_camera          → ekf_update_pos3   (3-DOF, pos only, absolute)
            (updates correct the shared state; the next VIO predict publishes)
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from tinynav.core.math_utils import msg2np, np2msg, pose_msg2np

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

# Process noise Q — error-state order: [δp(3), δv(3), δθ(3)]
Q_DIAG = np.array([
    0.001,  0.00,  0.00,     # δp   m²
    0.00,  0.010,  0.00,     # δv   (m/s)²
    0.00,  0.00,  0.0010,   # δθ   rad²
], dtype=np.float64)

# Measurement noise R.
# Robot odom now feeds the full 6-DOF pose6 update (all of x/y/z/roll/pitch/yaw),
# but a ground-robot base's z estimate is unreliable, so its observation noise
# is set very large — the Kalman gain on that row collapses to ~0, so z is
# effectively left to the other sources while x/y/roll/pitch/yaw still update.
R_ROBOT = np.diag([0.0001, 0.0001, 1.0e6,  0.00005, 0.00010, 0.00010])
R_WHEEL = np.diag([0.030, 0.030, 0.010,  0.005, 0.005, 0.030])
R_LIDAR = np.diag([0.020, 0.020, 0.010,  0.005, 0.005, 0.020])
R_QR    = np.diag([0.005, 0.005, 0.005,  0.002, 0.002, 0.005])
R_RTK   = np.diag([0.010, 0.010, 0.040])   # position-only 3×3

GATE: dict[str, float] = {
    'robot': 12.0,
    'wheel': 12.0,
    'lidar': 16.0,
    'qr':    10.0,
    'rtk':   16.0,
}

ROBOT_ODOM_TOPICS = tuple(
    topic.strip()
    for topic in os.environ.get('EKF_ROBOT_ODOM_TOPICS', '/robot/odom,/robot_odom').split(',')
    if topic.strip()
)

# base_link -> camera extrinsic, re-calibrated from
# debug_bags/rosbag2_2026_07_20-19_36_17 (/robot_odom vs
# /camera/camera/vio_image) using bag timestamps. In this bag the VIO header
# stamps live in a different time domain from /robot_odom, so hand-eye fitting
# must align on bag time rather than msg.header.stamp. /robot_odom reports
# T_odom_base, not T_world_camera like the other update sources — without this
# conjugation, base_link yaw leaks into apparent camera roll/pitch every time
# the robot turns.
T_BASE_CAM = np.array([
    [-0.01229927, -0.02659780,  0.99957055,  0.25667931],
    [-0.99991609, -0.00373790, -0.01240298, -0.00101519],
    [ 0.00406618, -0.99963923, -0.02654959,  0.03080751],
    [ 0.0,         0.0,         0.0,         1.0        ],
])

# Reject a SLAM predict step outright if the per-axis frame-to-frame delta
# exceeds this (m) — guards against VIO glitches/jumps corrupting predict,
# which has no Mahalanobis gate of its own (predict has no innovation/S).
SLAM_PREDICT_POS_GATE_M = 0.5

# Initial error-state covariance P0
P0_DIAG = np.array([
    1.0, 1.0, 0.5,    # δp
    0.5, 0.5, 0.2,    # δv
    0.1, 0.1, 0.2,    # δθ
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Observation matrices  (error-state space)
# ---------------------------------------------------------------------------

# Full 6-DOF: [δp_innov(3), δθ_innov(3)] observed from error state [δp, δv, δθ]
_H6 = np.zeros((6, 9))
_H6[0:3, 0:3] = np.eye(3)   # position
_H6[3:6, 6:9] = np.eye(3)   # orientation (δθ)

# Position-only (RTK)
_H3 = np.zeros((3, 9))
_H3[0:3, 0:3] = np.eye(3)


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class NominalState:
    p: np.ndarray   # (3,)  position in world frame
    v: np.ndarray   # (3,)  velocity in world frame
    q: np.ndarray   # (4,)  quaternion [x,y,z,w]  (scipy convention)


@dataclasses.dataclass
class EKFState:
    nom:   NominalState
    P:     np.ndarray   # (9,9) error-state covariance
    stamp: float        # seconds


# ---------------------------------------------------------------------------
# Pure math utilities
# ---------------------------------------------------------------------------

def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [ 0.0,   -v[2],  v[1]],
        [ v[2],   0.0,  -v[0]],
        [-v[1],   v[0],  0.0 ],
    ])


def _qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiply, scipy [x,y,z,w] convention."""
    return (Rotation.from_quat(q1) * Rotation.from_quat(q2)).as_quat()


def _qinv(q: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(q).inv().as_quat()


def _Rmat(q: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(q).as_matrix()


# ---------------------------------------------------------------------------
# EKF pure functions
# ---------------------------------------------------------------------------

def _predict_nominal(nom: NominalState, T_delta: np.ndarray,
                     dt: float) -> NominalState:
    Rq  = _Rmat(nom.q)
    dp  = T_delta[:3, 3]
    dq  = Rotation.from_matrix(T_delta[:3, :3]).as_quat()

    p_new = nom.p + Rq @ dp
    v_new = Rq @ dp / max(dt, 1e-4)
    q_new = _qmul(nom.q, dq)
    q_new = q_new / np.linalg.norm(q_new)
    return NominalState(p=p_new, v=v_new, q=q_new)


def _predict_F(nom: NominalState, T_delta: np.ndarray, dt: float) -> np.ndarray:
    """9×9 error-state Jacobian for the predict step."""
    Rq = _Rmat(nom.q)
    dp = T_delta[:3, 3]
    dR = T_delta[:3, :3]

    F = np.zeros((9, 9))
    F[0:3, 0:3] = np.eye(3)                          # δp  → δp
    F[0:3, 6:9] = -Rq @ _skew(dp)                    # δθ  → δp
    F[3:6, 6:9] = -Rq @ _skew(dp) / max(dt, 1e-4)   # δθ  → δv
    F[6:9, 6:9] = dR.T                               # δθ  → δθ
    return F


def ekf_predict(state: EKFState, T_delta: np.ndarray,
                Q: np.ndarray, dt: float, stamp: float | None = None) -> EKFState:
    nom_new = _predict_nominal(state.nom, T_delta, dt)
    F       = _predict_F(state.nom, T_delta, dt)
    P_new   = F @ state.P @ F.T + Q
    return EKFState(nom=nom_new, P=P_new,
                    stamp=state.stamp + dt if stamp is None else stamp)


def _apply_correction(nom: NominalState, dx: np.ndarray) -> NominalState:
    """Inject 9-dim error-state correction into nominal state."""
    p_new = nom.p + dx[0:3]
    v_new = nom.v + dx[3:6]
    dq    = Rotation.from_rotvec(dx[6:9]).as_quat()
    q_new = _qmul(nom.q, dq)
    q_new = q_new / np.linalg.norm(q_new)
    return NominalState(p=p_new, v=v_new, q=q_new)


def _ekf_update(state: EKFState, innov: np.ndarray, H: np.ndarray,
                R_noise: np.ndarray, gate: float) -> tuple[EKFState, bool]:
    S  = H @ state.P @ H.T + R_noise
    d2 = float(innov @ np.linalg.solve(S, innov))
    if d2 > gate:
        return state, False

    K   = state.P @ H.T @ np.linalg.inv(S)
    dx  = K @ innov
    IKH = np.eye(9) - K @ H
    # Joseph form: numerically stable even when K is imprecise
    P_new   = IKH @ state.P @ IKH.T + K @ R_noise @ K.T
    nom_new = _apply_correction(state.nom, dx)
    return EKFState(nom=nom_new, P=P_new, stamp=state.stamp), True


def ekf_update_pose6(state: EKFState, T_meas: np.ndarray,
                     R_noise: np.ndarray, gate: float) -> tuple[EKFState, bool]:
    """6-DOF update from a 4×4 SE3 measurement (T_world_camera)."""
    dp = T_meas[:3, 3] - state.nom.p

    q_meas = Rotation.from_matrix(T_meas[:3, :3]).as_quat()
    q_err  = _qmul(_qinv(state.nom.q), q_meas)
    if q_err[3] < 0:   # enforce short-path (scalar part ≥ 0)
        q_err = -q_err
    dtheta = Rotation.from_quat(q_err).as_rotvec()

    innov = np.concatenate([dp, dtheta])
    return _ekf_update(state, innov, _H6, R_noise, gate)


def ekf_update_pos3(state: EKFState, p_meas: np.ndarray,
                    R_noise: np.ndarray, gate: float) -> tuple[EKFState, bool]:
    """Position-only update (e.g. RTK GPS)."""
    innov = p_meas - state.nom.p
    return _ekf_update(state, innov, _H3, R_noise, gate)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _T_from_nominal(nom: NominalState) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _Rmat(nom.q)
    T[:3,  3] = nom.p
    return T


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _sec_to_stamp(t: float) -> TimeMsg:
    sec = int(t)
    nanosec = int(round((t - sec) * 1e9))
    return TimeMsg(sec=sec, nanosec=nanosec)


# ---------------------------------------------------------------------------
# ROS node
# ---------------------------------------------------------------------------

class EKFOdomNode(Node):
    def __init__(self):
        super().__init__('ekf_odom_node')

        self._state: EKFState | None = None
        self._Q = np.diag(Q_DIAG)

        self._last_slam_T:     np.ndarray | None = None
        self._last_slam_stamp: float | None      = None


        self._last_robot_raw: np.ndarray | None = None
        self._last_robot_nom: np.ndarray | None = None
        self._last_robot_stamp: float | None = None
        self._last_wheel_raw: np.ndarray | None = None
        self._last_wheel_nom: np.ndarray | None = None
        self._last_lidar_raw: np.ndarray | None = None
        self._last_lidar_nom: np.ndarray | None = None

        self.create_subscription(
            PoseStamped, '/camera/camera/vio_image', self._slam_20hz_cb, 100)
        self.create_subscription(
            PoseStamped, '/camera/camera/vio_100hz', self._slam_100hz_cb, 200)
        for topic in ROBOT_ODOM_TOPICS:
            self.create_subscription(Odometry, topic, self._robot_cb, 100)
        self.create_subscription(
            Odometry, '/wheel/odom_camera', self._wheel_cb,  100)
        self.create_subscription(
            Odometry, '/lidar/odom_camera', self._lidar_cb,   50)
        self.create_subscription(
            Odometry, '/qr/odom',           self._qr_cb,      10)
        self.create_subscription(
            Odometry, '/rtk/odom_camera',   self._rtk_cb,     10)

        self._pub        = self.create_publisher(Odometry, '/slam/odometry_fused',       10)
        self._pub_100hz  = self.create_publisher(Odometry, '/slam/odometry_fused_100hz', 10)
        self.get_logger().info(
            'ekf_odom_node ready  [error-state EKF, quaternion orientation]; '
            f'robot odom topics={ROBOT_ODOM_TOPICS}'
        )

    # ---- callbacks ----

    def _slam_20hz_cb(self, msg: PoseStamped) -> None:
        self._slam_predict(msg, 'slam-20hz', self._pub)

    def _slam_100hz_cb(self, msg: PoseStamped) -> None:
        self._slam_predict(msg, 'slam-100hz', self._pub_100hz)

    def _slam_predict(self, msg: PoseStamped, source: str, pub) -> None:
        """Predict step shared by both VIO rates. /camera/camera/vio_image
        (20Hz, optimized/corrected) and /camera/camera/vio_100hz
        (100Hz, IMU-propagated) are the same
        underlying trajectory at different rates/refinement"""
        T = pose_msg2np(msg)
        stamp = _stamp_to_sec(msg.header.stamp)

        if self._last_slam_T is None:
            self._last_slam_T     = T
            self._last_slam_stamp = stamp
            if self._state is None:
                self._init(T, stamp)
            return

        dt = stamp - self._last_slam_stamp
        if dt <= 0.0:
            if pub is self._pub and self._state is not None:
                self._publish(pub, msg.header.stamp)
            return

        T_delta               = np.linalg.inv(self._last_slam_T) @ T
        self._last_slam_T     = T
        self._last_slam_stamp = stamp

        if self._state is None:
            self._init(T, stamp)
            return

        dp = T_delta[:3, 3]
        if np.any(np.abs(dp) > SLAM_PREDICT_POS_GATE_M):
            self.get_logger().warn(
                f'[{source}] predict delta outlier rejected: dp={dp.tolist()} '
                f'(> {SLAM_PREDICT_POS_GATE_M} m per axis)', throttle_duration_sec=1.0)
            return

        self._state = ekf_predict(self._state, T_delta, self._Q, dt, stamp=stamp)
        self._publish(pub, msg.header.stamp)

    def _robot_cb(self, msg: Odometry) -> None:
        stamp = _stamp_to_sec(msg.header.stamp)
        if stamp <= 0.0:
            self.get_logger().warn(
                '[robot] odom with zero timestamp ignored',
                throttle_duration_sec=1.0,
            )
            return
        if self._last_robot_stamp is not None and stamp <= self._last_robot_stamp:
            if stamp < self._last_robot_stamp:
                self.get_logger().warn(
                    '[robot] odom timestamp moved backwards; reset delta reference',
                    throttle_duration_sec=1.0,
                )
                self._last_robot_raw = None
                self._last_robot_nom = None
                self._last_robot_stamp = stamp
            return
        self._last_robot_stamp = stamp
        self._update_pose6_delta(
            msg, R_ROBOT, GATE['robot'], 'robot', '_last_robot_raw', '_last_robot_nom',
            extrinsic=T_BASE_CAM)

    def _wheel_cb(self, msg: Odometry) -> None:
        self._update_pose6_delta(
            msg, R_WHEEL, GATE['wheel'], 'wheel', '_last_wheel_raw', '_last_wheel_nom')

    def _lidar_cb(self, msg: Odometry) -> None:
        self._update_pose6_delta(
            msg, R_LIDAR, GATE['lidar'], 'lidar', '_last_lidar_raw', '_last_lidar_nom')

    def _qr_cb(self, msg: Odometry) -> None:
        self._update_pose6(msg, R_QR, GATE['qr'], 'qr')

    def _rtk_cb(self, msg: Odometry) -> None:
        T, _ = msg2np(msg)
        stamp = _stamp_to_sec(msg.header.stamp)
        if self._state is None:
            self._init(T, stamp)
            return
        self._state, ok = ekf_update_pos3(
            self._state, T[:3, 3], R_RTK, GATE['rtk'])
        if not ok:
            self.get_logger().warn('rtk: outlier rejected',
                                   throttle_duration_sec=1.0)

    # ---- helpers ----

    def _update_pose6(self, msg: Odometry, R_noise: np.ndarray,
                      gate: float, source: str) -> None:
        T, _ = msg2np(msg)
        if self._state is None:
            self._init(T, _stamp_to_sec(msg.header.stamp))
            return
        self._state, ok = ekf_update_pose6(self._state, T, R_noise, gate)
        if not ok:
            self.get_logger().warn(
                f'[{source}] outlier rejected', throttle_duration_sec=1.0)

    def _update_pose6_delta(self, msg: Odometry, R_noise: np.ndarray, gate: float,
                            source: str, raw_attr: str, nom_attr: str,
                            extrinsic: np.ndarray | None = None) -> None:
        """Update using only the motion delta since this source's last reading,
        re-anchored onto the fused nominal pose at that time. Avoids trusting
        the source's own absolute origin/heading, which need not agree with
        the EKF's world frame.

        extrinsic, if given, is T_base_cam: this source publishes in its own
        rigid-body frame (e.g. base_link), not the camera frame the nominal
        state lives in, so its raw pose/delta is conjugated into camera frame
        before use — otherwise the source's own rotation axes leak into the
        wrong nominal-state axes (e.g. base_link yaw appearing as camera
        roll/pitch whenever the robot turns)."""
        T_raw = msg2np(msg)[0]
        if self._state is None:
            T_init = T_raw @ extrinsic if extrinsic is not None else T_raw
            self._init(T_init, _stamp_to_sec(msg.header.stamp))
            setattr(self, raw_attr, T_raw)
            setattr(self, nom_attr, _T_from_nominal(self._state.nom))
            return

        last_raw = getattr(self, raw_attr)
        if last_raw is None:
            setattr(self, raw_attr, T_raw)
            setattr(self, nom_attr, _T_from_nominal(self._state.nom))
            return

        T_delta = np.linalg.inv(last_raw) @ T_raw
        if extrinsic is not None:
            T_delta = np.linalg.inv(extrinsic) @ T_delta @ extrinsic
        T_meas  = getattr(self, nom_attr) @ T_delta

        self._state, ok = ekf_update_pose6(self._state, T_meas, R_noise, gate)
        if not ok:
            self.get_logger().warn(
                f'[{source}] outlier rejected', throttle_duration_sec=1.0)

        setattr(self, raw_attr, T_raw)
        setattr(self, nom_attr, _T_from_nominal(self._state.nom))

    def _init(self, T: np.ndarray, stamp: float) -> None:
        q   = Rotation.from_matrix(T[:3, :3]).as_quat()
        nom = NominalState(p=T[:3, 3].copy(), v=np.zeros(3), q=q)
        self._state = EKFState(nom=nom, P=np.diag(P0_DIAG), stamp=stamp)
        self.get_logger().info('EKF state initialized.')

    def _publish(self, pub, stamp_msg=None) -> None:
        """Publish only from the VIO predict tick that owns this output topic.
        This keeps /slam/odometry_fused stamped exactly like /camera/camera/vio_image
        and /slam/odometry_fused_100hz stamped like /camera/camera/vio_100hz."""
        if self._state is None:
            return
        T = _T_from_nominal(self._state.nom)
        stamp = _sec_to_stamp(self._state.stamp) if stamp_msg is None else stamp_msg
        msg = np2msg(T, stamp, 'world', 'camera', velocity=self._state.nom.v)
        pub.publish(msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = EKFOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
