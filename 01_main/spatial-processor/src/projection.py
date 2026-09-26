"""
Depthマップ -> 3D点群への逆投影(10章)、ToFによる絶対スケール補正(11章)、
IMUによる姿勢回転(12章)を行う。
"""
import numpy as np


def backproject(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float, valid_region):
    """
    depth: (H,W) float32 相対深度
    valid_region: (x0,y0,x1,y1) letterboxの有効領域(黒帯を除外)
    戻り値: points (N,3) float32 [カメラ座標系, スケール未補正]
    """
    x0, y0, x1, y1 = valid_region
    us, vs = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    z = depth[y0:y1, x0:x1]

    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return points.astype(np.float32)


def estimate_scale(depth: np.ndarray, valid_region, tof_mm: float) -> float:
    """
    ToF測定領域(画像中央付近)の相対深度から代表値を求め、
    ToF実測値とのスケール係数 K = S_tof / D_reference を計算する(仕様書11章)。
    外れ値除去には中央絶対偏差(MAD)ベースのフィルタを使う。
    """
    if tof_mm <= 0:
        return 1.0

    x0, y0, x1, y1 = valid_region
    center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
    half = max(2, (x1 - x0) // 20)  # 中央付近の小領域をToF測定域とみなす

    y_lo, y_hi = max(y0, center_y - half), min(y1, center_y + half)
    x_lo, x_hi = max(x0, center_x - half), min(x1, center_x + half)
    patch = depth[y_lo:y_hi, x_lo:x_hi]
    if patch.size == 0:
        return 1.0

    med = float(np.median(patch))
    mad = float(np.median(np.abs(patch - med))) + 1e-6
    inliers = patch[np.abs(patch - med) < 3 * mad]
    d_reference = float(np.median(inliers)) if inliers.size > 0 else med

    if d_reference <= 1e-6:
        return 1.0

    s_tof_m = tof_mm / 1000.0
    return s_tof_m / d_reference


def rotation_matrix(pitch_deg: float, roll_deg: float, yaw_deg: float) -> np.ndarray:
    """IMUのpitch/roll/yaw(度)から回転行列を作成する(カメラ座標系 -> ローカルワールド座標系)"""
    p, r, y = np.radians([pitch_deg, roll_deg, yaw_deg])

    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    ry = np.array([[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])

    return (rz @ ry @ rx).astype(np.float32)


def to_local_world(points_cam: np.ndarray, pitch: float, roll: float, yaw: float) -> np.ndarray:
    """カメラ座標系の点群を、ドローン基準のローカルワールド座標系へ変換する(13章)"""
    r = rotation_matrix(pitch, roll, yaw)
    return points_cam @ r.T
