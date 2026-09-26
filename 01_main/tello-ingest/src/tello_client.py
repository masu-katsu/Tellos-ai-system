"""
Ryze Tello との低レベル通信を担当する。
  - コマンド送受信 (UDP 8889)
  - 状態(pitch/roll/yaw/ToF/battery)受信 (UDP 8890)
  - 映像ストリームのUDP URLを提供 (11111, H.264 RAW)
"""
import logging
import socket
import threading
import time

logger = logging.getLogger("tello_client")

CMD_PORT = 8889
STATE_PORT = 8890
VIDEO_PORT = 11111
RESPONSE_TIMEOUT_S = 5.0


class TelloState:
    """最新のテレメトリを保持する。スレッド間はロックで簡易的に保護する。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        self.tof_mm = 0
        self.battery = 0
        self.updated_at = 0.0

    def update_from_line(self, line: str):
        # Tello state format 例: "pitch:0;roll:0;yaw:0;...;tof:100;...;bat:82;..."
        fields = {}
        for item in line.strip().rstrip(";").split(";"):
            if ":" not in item:
                continue
            k, v = item.split(":", 1)
            fields[k] = v

        with self._lock:
            try:
                self.pitch = float(fields.get("pitch", self.pitch))
                self.roll = float(fields.get("roll", self.roll))
                self.yaw = float(fields.get("yaw", self.yaw))
                self.tof_mm = int(float(fields.get("tof", self.tof_mm)))
                self.battery = int(float(fields.get("bat", self.battery)))
                self.updated_at = time.time()
            except ValueError:
                logger.warning("state parse error: %r", line)

    def snapshot(self):
        with self._lock:
            return {
                "pitch": self.pitch,
                "roll": self.roll,
                "yaw": self.yaw,
                "tof_mm": self.tof_mm,
                "battery": self.battery,
                "updated_at": self.updated_at,
            }


class TelloClient:
    def __init__(self, tello_ip: str):
        self.tello_ip = tello_ip
        self.state = TelloState()

        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cmd_sock.bind(("", CMD_PORT))
        self._cmd_sock.settimeout(RESPONSE_TIMEOUT_S)

        self._state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._state_sock.bind(("", STATE_PORT))

        self._running = False
        self._state_thread = None

    def send_command(self, cmd: str) -> str:
        logger.info("-> %s", cmd)
        self._cmd_sock.sendto(cmd.encode("utf-8"), (self.tello_ip, CMD_PORT))
        try:
            data, _ = self._cmd_sock.recvfrom(1024)
            resp = data.decode("utf-8", errors="ignore").strip()
            logger.info("<- %s", resp)
            return resp
        except socket.timeout:
            logger.warning("command timeout: %s", cmd)
            return ""

    def connect(self):
        self.send_command("command")
        self.send_command("streamon")
        self._running = True
        self._state_thread = threading.Thread(target=self._state_loop, daemon=True)
        self._state_thread.start()

    def _state_loop(self):
        while self._running:
            try:
                data, _ = self._state_sock.recvfrom(1024)
                self.state.update_from_line(data.decode("utf-8", errors="ignore"))
            except OSError:
                if self._running:
                    logger.exception("state socket error")

    def video_url(self) -> str:
        return f"udp://0.0.0.0:{VIDEO_PORT}"

    def close(self):
        self._running = False
        try:
            self.send_command("streamoff")
        except Exception:
            logger.warning("streamoff failed (ignored during shutdown)")
        self._cmd_sock.close()
        self._state_sock.close()
