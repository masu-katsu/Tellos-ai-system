"""
spatial-processor-service エントリポイント(仕様書9〜17章)。

Depthマップ(Shared Memory) + IMU + ToF + カメラ内部パラメータ から
  3D点群への逆投影 -> 絶対スケール補正 -> 姿勢補正 -> ボクセル化
を行い、UDPバイナリでUnityへ送信する。常に最新フレームのみを処理する。
"""
import logging
import os
import sys
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from common.logging_config import setup_logging  # noqa: E402
from common.shared_depth import DEPTH_H, DEPTH_W, SharedDepthBuffer  # noqa: E402

from projection import backproject, estimate_scale, to_local_world  # noqa: E402
from udp_sender import UnityUdpSender  # noqa: E402
from voxelize import voxelize  # noqa: E402

SERVICE_NAME = "spatial-processor"
setup_logging(SERVICE_NAME)
logger = logging.getLogger(SERVICE_NAME)

# カメラ内部パラメータ(要キャリブレーション。420x420換算の暫定値)
FX = float(os.environ.get("CAM_FX", 350.0))
FY = float(os.environ.get("CAM_FY", 350.0))
CX = float(os.environ.get("CAM_CX", DEPTH_W / 2))
CY = float(os.environ.get("CAM_CY", DEPTH_H / 2))

VOXEL_SIZE_M = float(os.environ.get("VOXEL_SIZE_M", 0.07))
MAX_VOXELS = int(os.environ.get("MAX_VOXELS", 4000))

UNITY_HOST = os.environ.get("UNITY_HOST", "127.0.0.1")
UNITY_PORT = int(os.environ.get("UNITY_PORT", 9999))

LOG_EVERY_N_FRAMES = 30
POLL_INTERVAL_S = 0.001
SHM_WAIT_S = 1.0


def wait_for_input_shm(retries: int = 30):
    """ai-inferenceが先にShared Memoryを作成しているのを待つ"""
    for _ in range(retries):
        try:
            return SharedDepthBuffer(create=False)
        except FileNotFoundError:
            logger.info("waiting for ai-inference shared memory...")
            time.sleep(SHM_WAIT_S)
    raise RuntimeError("ai-inferenceのShared Memoryが見つかりません")


def main():
    in_shm = wait_for_input_shm()
    sender = UnityUdpSender(UNITY_HOST, UNITY_PORT)

    last_frame_id = -1
    logger.info("spatial-processor started. sending to %s:%d", UNITY_HOST, UNITY_PORT)

    try:
        while True:
            frame = in_shm.read()
            if frame["frame_id"] == 0 or frame["frame_id"] == last_frame_id:
                time.sleep(POLL_INTERVAL_S)
                continue
            last_frame_id = frame["frame_id"]

            depth = np.frombuffer(frame["depth_bytes"], dtype=np.float32).reshape(DEPTH_H, DEPTH_W)

            scale = estimate_scale(depth, frame["valid_region"], frame["tof_mm"])
            points_cam = backproject(depth, FX, FY, CX, CY, frame["valid_region"])
            points_cam *= scale

            points_world = to_local_world(points_cam, frame["pitch"], frame["roll"], frame["yaw"])
            voxels = voxelize(points_world, voxel_size_m=VOXEL_SIZE_M, max_voxels=MAX_VOXELS)

            n_packets = sender.send(
                frame_id=frame["frame_id"],
                pitch=frame["pitch"], roll=frame["roll"], yaw=frame["yaw"],
                voxels=voxels.tolist(),
            )

            if frame["frame_id"] % LOG_EVERY_N_FRAMES == 0:
                logger.info(
                    "frame=%d voxels=%d packets=%d scale=%.3f",
                    frame["frame_id"], len(voxels), n_packets, scale,
                )
    finally:
        in_shm.close()
        sender.close()


if __name__ == "__main__":
    main()
