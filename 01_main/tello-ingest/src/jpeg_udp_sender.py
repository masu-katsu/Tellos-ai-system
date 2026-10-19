"""最新のRGBフレームをJPEG化し、Unity向けUDPパケットへ分割して送信する。

パケットヘッダーは little-endian ``<4sIHHI``:
magic ``TJPG``、frame_id(uint32)、fragment_index(uint16, 0始まり)、
fragment_count(uint16)、jpeg_size(uint32)。ヘッダーを含むdatagramは最大1200 byte。
Unity側はframe_idごとに断片を集め、jpeg_size分を復元してJPEGデコードする。
"""
import logging
import socket
import struct
import threading

import cv2

logger = logging.getLogger("jpeg_udp_sender")

MAGIC = b"TJPG"
HEADER_FMT = "<4sIHHI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
MAX_DATAGRAM_SIZE = 1200
JPEG_CHUNK_SIZE = MAX_DATAGRAM_SIZE - HEADER_SIZE
DEFAULT_PORT = 9998
DEFAULT_JPEG_QUALITY = 80


class LatestJpegUdpSender:
    def __init__(self, host: str, port: int = DEFAULT_PORT, jpeg_quality: int = DEFAULT_JPEG_QUALITY):
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")

        self.addr = (host, port)
        self.jpeg_quality = jpeg_quality
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._condition = threading.Condition()
        self._latest_frame = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="jpeg-udp-sender", daemon=True)
        self._thread.start()

    def submit(self, frame_id: int, rgb):
        with self._condition:
            if self._closed:
                return
            self._latest_frame = (frame_id, rgb.copy())
            self._condition.notify()

    def close(self):
        with self._condition:
            self._closed = True
            self._latest_frame = None
            self._condition.notify_all()
        self._thread.join()

    def _run(self):
        try:
            while True:
                with self._condition:
                    while self._latest_frame is None and not self._closed:
                        self._condition.wait()
                    if self._closed:
                        return
                    frame_id, rgb = self._latest_frame
                    self._latest_frame = None

                try:
                    self._send_frame(frame_id, rgb)
                except (OSError, cv2.error, ValueError):
                    logger.exception("JPEG frame send failed: frame_id=%d", frame_id)
        finally:
            self.sock.close()

    def _send_frame(self, frame_id: int, rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            raise ValueError("OpenCV failed to encode JPEG")

        jpeg = encoded.tobytes()
        fragment_count = (len(jpeg) + JPEG_CHUNK_SIZE - 1) // JPEG_CHUNK_SIZE
        if fragment_count > 0xFFFF:
            raise ValueError(f"JPEG is too large to fragment: {len(jpeg)} bytes")

        for fragment_index in range(fragment_count):
            start = fragment_index * JPEG_CHUNK_SIZE
            chunk = jpeg[start:start + JPEG_CHUNK_SIZE]
            header = struct.pack(
                HEADER_FMT,
                MAGIC,
                frame_id & 0xFFFFFFFF,
                fragment_index,
                fragment_count,
                len(jpeg),
            )
            self.sock.sendto(header + chunk, self.addr)
