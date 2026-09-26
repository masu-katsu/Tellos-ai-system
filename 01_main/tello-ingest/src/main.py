"""
tello-ingest-service エントリポイント(仕様書4〜6章)。

Tello -> H.264デコード -> letterbox(420x420) -> IMU/ToF付与 -> Shared Memory書き込み。
このコンテナより後段は、Telloの通信仕様を直接意識しなくてよい。
常に最新フレームのみを保持し、処理が追いつかない場合は古いフレームを破棄する。
"""
import logging
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # commonをimport可能にする

from common.logging_config import setup_logging  # noqa: E402
from common.shared_frame import SharedFrameBuffer  # noqa: E402

from letterbox import letterbox  # noqa: E402
from tello_client import TelloClient  # noqa: E402
from video_decoder import frame_generator  # noqa: E402

SERVICE_NAME = "tello-ingest"
setup_logging(SERVICE_NAME)
logger = logging.getLogger(SERVICE_NAME)

TELLO_IP = os.environ.get("TELLO_IP", "192.168.10.1")
LOG_EVERY_N_FRAMES = 30
RETRY_INTERVAL_S = 2.0


def run():
    client = TelloClient(TELLO_IP)
    client.connect()

    state = client.state.snapshot()
    logger.info("Tello connected. battery=%s%%", state.get("battery"))

    shm = SharedFrameBuffer(create=True)
    frame_id = 0

    try:
        for rgb, _pts in frame_generator(client.video_url()):
            frame_id += 1
            canvas, valid_region = letterbox(rgb)
            state = client.state.snapshot()

            shm.write(
                frame_id=frame_id,
                timestamp=time.time(),
                rgb_bytes=canvas.tobytes(),
                pitch=state["pitch"],
                roll=state["roll"],
                yaw=state["yaw"],
                tof_mm=state["tof_mm"],
                valid_region=valid_region,
            )

            if frame_id % LOG_EVERY_N_FRAMES == 0:
                logger.info(
                    "frame=%d pitch=%.1f roll=%.1f yaw=%.1f tof=%dmm battery=%d%%",
                    frame_id, state["pitch"], state["roll"], state["yaw"],
                    state["tof_mm"], state["battery"],
                )
    finally:
        client.close()
        shm.unlink()


def main():
    # Tello未接続/映像未受信時は一定間隔でリトライする(電源投入直後やWiFi再接続を想定)
    while True:
        try:
            run()
            break
        except KeyboardInterrupt:
            break
        except Exception:
            logger.exception("tello-ingest crashed - retrying in %.1fs", RETRY_INTERVAL_S)
            time.sleep(RETRY_INTERVAL_S)


if __name__ == "__main__":
    main()
