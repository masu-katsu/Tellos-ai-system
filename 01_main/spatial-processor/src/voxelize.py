"""
3D点群をボクセル(5〜10cm単位)に量子化し、重複を除去する(仕様書14章)。

    50,000 points -> 4,000 voxels

のように、Unityへ送るデータ量を大きく圧縮する。
"""
import numpy as np


def voxelize(points: np.ndarray, voxel_size_m: float = 0.07, max_voxels: int = 4000) -> np.ndarray:
    """
    points: (N,3) float32 [m], ローカルワールド座標系
    戻り値: voxel中心座標 (M,3) float32 (M <= max_voxels)
    """
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)

    # 無効値(NaN/inf)や近すぎ/遠すぎる点を除外
    finite = np.isfinite(points).all(axis=1)
    plausible_range = (points[:, 2] > 0.05) & (points[:, 2] < 20.0)
    points = points[finite & plausible_range]
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)

    idx = np.floor(points / voxel_size_m).astype(np.int64)
    _, unique_indices = np.unique(idx, axis=0, return_index=True)
    voxel_centers = (idx[unique_indices].astype(np.float32) + 0.5) * voxel_size_m

    if voxel_centers.shape[0] > max_voxels:
        # 密度が高すぎる場合はランダム間引き(近接ボクセルを均等に落とす)
        rng = np.random.default_rng()
        choice = rng.choice(voxel_centers.shape[0], max_voxels, replace=False)
        voxel_centers = voxel_centers[choice]

    return voxel_centers
