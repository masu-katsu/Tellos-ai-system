"""
ai-inference -> spatial-processor 間で「常に最新のDepthマップだけ」を受け渡す共有メモリバッファ。
shared_frame.py と同じ設計方針(上書き式・キューなし)。
"""
import struct
from multiprocessing import shared_memory

try:
    import posix_ipc
except ImportError:  # Windows host / non-POSIX runtime fallback
    posix_ipc = None

DEPTH_W = 420
DEPTH_H = 420
DEPTH_BYTES = DEPTH_W * DEPTH_H * 4  # float32

HEADER_FMT = "<Id3fH4H"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

SHM_NAME = "tello_latest_depth"
SEM_NAME = "/tello_depth_sem"


class _NoopSemaphore:
    def acquire(self):
        return None

    def release(self):
        return None

    def unlink(self):
        return None


def _open_semaphore(name: str, create: bool = False, initial_value: int = 1):
    if posix_ipc is None:
        return _NoopSemaphore()
    if create:
        return posix_ipc.Semaphore(name, flags=posix_ipc.O_CREAT, initial_value=initial_value)
    return posix_ipc.Semaphore(name)


class SharedDepthBuffer:
    """1フレーム分のDepthマップ(float32, 420x420)+テレメトリを保持する共有メモリ。"""

    def __init__(self, create: bool):
        size = HEADER_SIZE + DEPTH_BYTES
        if create:
            self._unlink_stale()
            self.shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=size)
            self.sem = _open_semaphore(SEM_NAME, create=True, initial_value=1)
            self.shm.buf[:HEADER_SIZE] = struct.pack(HEADER_FMT, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0)
        else:
            self.shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
            self.sem = _open_semaphore(SEM_NAME, create=False)

    @staticmethod
    def _unlink_stale():
        try:
            shared_memory.SharedMemory(name=SHM_NAME).unlink()
        except FileNotFoundError:
            pass
        if posix_ipc is None:
            return
        try:
            posix_ipc.Semaphore(SEM_NAME).unlink()
        except posix_ipc.ExistentialError:
            pass

    def write(self, frame_id, timestamp, depth_f32_bytes, pitch, roll, yaw, tof_mm, valid_region):
        assert len(depth_f32_bytes) == DEPTH_BYTES, "Depthバッファサイズが420x420xfloat32と一致しません"
        x0, y0, x1, y1 = valid_region
        header = struct.pack(
            HEADER_FMT, frame_id & 0xFFFFFFFF, timestamp, pitch, roll, yaw, int(tof_mm), x0, y0, x1, y1
        )
        if self.sem is not None:
            self.sem.acquire()
        try:
            self.shm.buf[:HEADER_SIZE] = header
            self.shm.buf[HEADER_SIZE:HEADER_SIZE + DEPTH_BYTES] = depth_f32_bytes
        finally:
            if self.sem is not None:
                self.sem.release()

    def read(self):
        if self.sem is not None:
            self.sem.acquire()
        try:
            header = bytes(self.shm.buf[:HEADER_SIZE])
            depth = bytes(self.shm.buf[HEADER_SIZE:HEADER_SIZE + DEPTH_BYTES])
        finally:
            if self.sem is not None:
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
            "depth_bytes": depth,
        }

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()
        if self.sem is None:
            return
        try:
            self.sem.unlink()
        except posix_ipc.ExistentialError:
            pass
