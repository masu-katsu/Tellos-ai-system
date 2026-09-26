"""
Tello の生H.264 UDPストリームをデコードして numpy(RGB) フレームを生成する。
PyAV(ffmpeg binding) を使用し、低遅延オプションでコンテナを開く。
"""
import logging

import av

logger = logging.getLogger("video_decoder")

_LOW_DELAY_OPTIONS = {
    "fflags": "nobuffer",
    "flags": "low_delay",
    "probesize": "32",
    "max_delay": "0",
}


def frame_generator(video_url: str):
    """
    video_url: 'udp://0.0.0.0:11111'
    yields: (rgb_ndarray, pts_seconds)
    """
    logger.info("opening video stream: %s", video_url)
    container = av.open(video_url, options=_LOW_DELAY_OPTIONS, timeout=10)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    try:
        for packet in container.demux(stream):
            for frame in packet.decode():
                img = frame.to_ndarray(format="rgb24")
                yield img, float(frame.time or 0.0)
    finally:
        container.close()
        logger.info("video stream closed")
