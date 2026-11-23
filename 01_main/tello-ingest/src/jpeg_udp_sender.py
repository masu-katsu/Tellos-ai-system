"""最新のRGBフレームを4:3のJPEGにし、Unity向けUDPパケットへ分割して送信する。

Unity側との通信契約(4.3 映像 datagram / UDP 9998)に準拠する。

パケット(リトルエンディアン):
  ヘッダー 10 byte ``<IHHH``
    Frame_ID      uint32  ボクセル(UDP 9999)と同じ入力フレーム由来の値
    Packet_ID     uint16  0始まり
    Total_Packets uint16  1〜219
    Payload_Len   uint16  1〜1200
  本体: JPEGの連続した断片(Payload_Len byte)

分割ルール:
  - datagram長は厳密に 10 + Payload_Len
  - 最終パケット以外の Payload_Len は 1200 byte 固定、最終パケットは 1〜1200 byte
  - 再構成後のJPEGは 256 KiB 以下(超える場合は品質を下げて再エンコードする)

画像の条件(Unity側で検証される):
  - 幅・高さは 1〜640 x 1〜480
  - アスペクト比は 4:3 (このモジュールでは正確に 4:3 とする)
  - 深度推定用の420x420レターボックス画像ではなく、元フレームを4:3に整えて送る
"""
import logging
import socket
import struct
import threading

import cv2

logger = logging.getLogger("jpeg_udp_sender")

HEADER_FMT = "<IHHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 10 byte
MAX_PAYLOAD_SIZE = 1200
MAX_PACKETS_PER_FRAME = 219
MAX_JPEG_BYTES = 262144  # 256 KiB

MAX_VIDEO_WIDTH = 640
MAX_VIDEO_HEIGHT = 480

DEFAULT_PORT = 9998
DEFAULT_JPEG_QUALITY = 80
DEFAULT_VIDEO_WIDTH = 480
DEFAULT_VIDEO_HEIGHT = 360

MIN_JPEG_QUALITY = 30
JPEG_QUALITY_STEP = 10


def fit_to_4x3(rgb, width: int, height: int):
    """
    元フレームを中央クロップで4:3に整え、(width x height) に縮小して返す。
    引き伸ばして歪ませないよう、縦横比が違う場合は余分な部分を切り落とす。
    """
    h, w = rgb.shape[:2]
    if w * 3 > h * 4:
        # 4:3より横長 -> 左右を切り落とす
        new_w = max(1, h * 4 // 3)
        x0 = (w - new_w) // 2
        rgb = rgb[:, x0:x0 + new_w]
    elif w * 3 < h * 4:
        # 4:3より縦長 -> 上下を切り落とす
        new_h = max(1, w * 3 // 4)
        y0 = (h - new_h) // 2
        rgb = rgb[y0:y0 + new_h, :]

    if rgb.shape[1] != width or rgb.shape[0] != height:
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return rgb


def split_jpeg_into_datagrams(frame_id: int, jpeg: bytes):
    """JPEGバイト列を、契約どおりのヘッダー付きdatagramのリストにする。"""
    if not 1 <= len(jpeg) <= MAX_JPEG_BYTES:
        raise ValueError(f"JPEG size out of range: {len(jpeg)} bytes")

    total = (len(jpeg) + MAX_PAYLOAD_SIZE - 1) // MAX_PAYLOAD_SIZE
    if total > MAX_PACKETS_PER_FRAME:
        raise ValueError(f"JPEG needs too many packets: {total}")

    datagrams = []
    for packet_id in range(total):
        start = packet_id * MAX_PAYLOAD_SIZE
        chunk = jpeg[start:start + MAX_PAYLOAD_SIZE]
        header = struct.pack(HEADER_FMT, frame_id & 0xFFFFFFFF, packet_id, total, len(chunk))
        datagrams.append(header + chunk)
    return datagrams


class LatestJpegUdpSender:
    def __init__(
        self,
        host: str,
        port: int = DEFAULT_PORT,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        width: int = DEFAULT_VIDEO_WIDTH,
        height: int = DEFAULT_VIDEO_HEIGHT,
    ):
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if not (1 <= width <= MAX_VIDEO_WIDTH and 1 <= height <= MAX_VIDEO_HEIGHT):
            raise ValueError(f"video size must be within {MAX_VIDEO_WIDTH}x{MAX_VIDEO_HEIGHT}")
        if width * 3 != height * 4:
            raise ValueError("video size must be exactly 4:3")

        self.addr = (host, port)
        self.jpeg_quality = jpeg_quality
        self.width = width
        self.height = height
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._condition = threading.Condition()
        self._latest_frame = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="jpeg-udp-sender", daemon=True)
        self._thread.start()

    def submit(self, frame_id: int, rgb):
        """
        最新フレームを渡す(古い未送信フレームは上書きして破棄する)。
        rgb は元フレーム(RGB, HxWx3)。呼び出し側は渡した配列を後から書き換えないこと
        (video_decoder は毎フレーム新しい配列を返すのでコピーしない)。
        """
        with self._condition:
            if self._closed:
                return
            self._latest_frame = (frame_id, rgb)
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

    def _encode(self, rgb) -> bytes:
        """4:3に整えてJPEG化する。256 KiBを超える場合は品質を下げて再エンコードする。"""
        bgr = cv2.cvtColor(fit_to_4x3(rgb, self.width, self.height), cv2.COLOR_RGB2BGR)

        quality = self.jpeg_quality
        while True:
            ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise ValueError("OpenCV failed to encode JPEG")
            jpeg = encoded.tobytes()
            if len(jpeg) <= MAX_JPEG_BYTES:
                return jpeg
            if quality <= MIN_JPEG_QUALITY:
                raise ValueError(f"JPEG is too large even at quality {quality}: {len(jpeg)} bytes")
            quality = max(MIN_JPEG_QUALITY, quality - JPEG_QUALITY_STEP)

    def _send_frame(self, frame_id: int, rgb):
        jpeg = self._encode(rgb)
        for datagram in split_jpeg_into_datagrams(frame_id, jpeg):
            self.sock.sendto(datagram, self.addr)
