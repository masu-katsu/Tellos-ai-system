"""
tello-ingest -> ai-inference 間で「常に最新の1フレームだけ」を受け渡すための共有メモリバッファ。

複数コンテナ間で multiprocessing.shared_memory / posix_ipc を使うには、
docker-compose 側で IPC namespace を共有しておく必要がある
  - tello-ingest: ipc: shareable
  - ai-inference / spatial-processor: ipc: "service:tello-ingest"

設計方針(仕様書 5〜6章):
  - キューは持たない。書き込みは常に上書き。
  - 読み出し側は常に「その時点で一番新しいフレーム」を受け取る。
"""
import struct
from multiprocessing import shared_memory

import posix_ipc

FRAME_W = 420
FRAME_H = 420
FRAME_CHANNELS = 3
FRAME_BYTES = FRAME_W * FRAME_H * FRAME_CHANNELS

# frame_id(u32) timestamp(f64) pitch,roll,yaw(f32 x3) tof_mm(u16) valid_region(u16 x4)
HEADER_FMT = "<Id3fH4H"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

SHM_NAME = "tello_latest_frame"
SEM_NAME = "/tello_frame_sem"


class SharedFrameBuffer:
    """1フレーム分のRGB+テレメトリを保持する共有メモリ。上書き式・常に最新のみ保持する。"""

    def __init__(self, create: bool):
        size = HEADER_SIZE + FRAME_BYTES
        if create:
            self._unlink_stale()
            self.shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=size)
            self.sem = posix_ipc.Semaphore(SEM_NAME, flags=posix_ipc.O_CREAT, initial_value=1)
            # frame_id=0 で初期化しておく(未書き込み状態を表す)
            self.shm.buf[:HEADER_SIZE] = struct.pack(HEADER_FMT, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0)
        else:
            self.shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
            self.sem = posix_ipc.Semaphore(SEM_NAME)

    @staticmethod
    def _unlink_stale():
        try:
            shared_memory.SharedMemory(name=SHM_NAME).unlink()
        except FileNotFoundError:
            pass
        try:
            posix_ipc.Semaphore(SEM_NAME).unlink()
        except posix_ipc.ExistentialError:
            pass

    def write(self, frame_id, timestamp, rgb_bytes, pitch, roll, yaw, tof_mm, valid_region):
        assert len(rgb_bytes) == FRAME_BYTES, "RGBバッファサイズが420x420x3と一致しません"
        x0, y0, x1, y1 = valid_region
        header = struct.pack(
            HEADER_FMT, frame_id & 0xFFFFFFFF, timestamp, pitch, roll, yaw, int(tof_mm), x0, y0, x1, y1
        )
        self.sem.acquire()
        try:
            self.shm.buf[:HEADER_SIZE] = header
            self.shm.buf[HEADER_SIZE:HEADER_SIZE + FRAME_BYTES] = rgb_bytes
        finally:
            self.sem.release()

    def read(self):
        self.sem.acquire()
        try:
            header = bytes(self.shm.buf[:HEADER_SIZE])
            rgb = bytes(self.shm.buf[HEADER_SIZE:HEADER_SIZE + FRAME_BYTES])
        finally:
            self.sem.release()
        frame_id, timestamp, pitch, roll, yaw, tof_mm, x0, y0, x1, y1 = struct.unpack(HEADER_FMT, header)
        return {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "pitch": pitch,
            "roll": roll,
            "yaw": yaw,
            "tof_mm": tof_mm,
            "valid_region": (x0, y0, x1, y1),
            "rgb": rgb,
        }

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()
        try:
            self.sem.unlink()
        except posix_ipc.ExistentialError:
            pass
