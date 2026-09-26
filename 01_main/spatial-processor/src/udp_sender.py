"""
ボクセルデータをUDPバイナリでUnityへ送信する(仕様書16章)。
"""
import logging
import os
import socket
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.udp_protocol import pack_voxel_frame  # noqa: E402

logger = logging.getLogger("udp_sender")


class UnityUdpSender:
    def __init__(self, host: str, port: int):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, frame_id: int, pitch: float, roll: float, yaw: float, voxels) -> int:
        """voxels: [(x,y,z), ...] を送信し、送信したパケット数を返す"""
        packets = pack_voxel_frame(frame_id, pitch, roll, yaw, voxels)
        for pkt in packets:
            self.sock.sendto(pkt, self.addr)
        return len(packets)

    def close(self):
        self.sock.close()
