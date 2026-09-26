"""
ai-inference-service エントリポイント(仕様書7〜8章)。

Shared Memory(RGBフレーム)を読み取り、Depth Anything V2 (TensorRT) で相対深度を推定し、
Shared Memory(Depthマップ)に書き込む。tello-ingestと同様、常に最新フレームのみを処理する。
"""
import logging
import os
import sys
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from common.logging_config import setup_logging  # noqa: E402
from common.shared_depth import SharedDepthBuffer  # noqa: E402
from common.shared_frame import FRAME_H, FRAME_W, SharedFrameBuffer  # noqa: E402

from depth_engine import DepthEngine  # noqa: E402

SERVICE_NAME = "ai-inference"
setup_logging(SERVICE_NAME)
logger = logging.getLogger(SERVICE_NAME)

ENGINE_PATH = os.environ.get("DEPTH_ENGINE_PATH", "/models/depth_anything_v2_vits.engine")
LOG_EVERY_N_FRAMES = 30
POLL_INTERVAL_S = 0.001
SHM_WAIT_S = 1.0


def wait_for_input_shm(retries: int = 30):
    """tello-ingestが先にShared Memoryを作成しているのを待つ"""
    for _ in range(retries):
        try:
            return SharedFrameBuffer(create=False)
        except FileNotFoundError:
            logger.info("waiting for tello-ingest shared memory...")
            time.sleep(SHM_WAIT_S)
    raise RuntimeError("tello-ingestのShared Memoryが見つかりません")


def main():
    engine = DepthEngine(ENGINE_PATH, input_size=FRAME_W)

    in_shm = wait_for_input_shm()
    out_shm = SharedDepthBuffer(create=True)

    last_frame_id = -1
    logger.info("ai-inference started. waiting for frames...")

    try:
        while True:
            frame = in_shm.read()
            if frame["frame_id"] == 0 or frame["frame_id"] == last_frame_id:
                time.sleep(POLL_INTERVAL_S)
                continue
            last_frame_id = frame["frame_id"]

            rgb = np.frombuffer(frame["rgb"], dtype=np.uint8).reshape(FRAME_H, FRAME_W, 3)

            t0 = time.time()
            depth = engine.infer(rgb)
            infer_ms = (time.time() - t0) * 1000

            out_shm.write(
                frame_id=frame["frame_id"],
                timestamp=frame["timestamp"],
                depth_f32_bytes=depth.astype(np.float32).tobytes(),
                pitch=frame["pitch"],
                roll=frame["roll"],
                yaw=frame["yaw"],
                tof_mm=frame["tof_mm"],
                valid_region=frame["valid_region"],
            )

            if frame["frame_id"] % LOG_EVERY_N_FRAMES == 0:
                logger.info("frame=%d infer=%.1fms", frame["frame_id"], infer_ms)
    finally:
        in_shm.close()
        out_shm.unlink()


if __name__ == "__main__":
    main()
