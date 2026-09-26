"""
spatial-processor -> Unity へ送るUDPバイナリプロトコル(仕様書16〜17章)。
JSONは使わずBinary固定フォーマット。1パケットあたり約1.2KB以下に収める。

パケットフォーマット:
  Header
    Frame_ID       uint32
    Packet_ID      uint16
    Total_Packets  uint16
    Voxel_Count    uint16
    Drone_Pitch    int16   (度 x100 の固定小数)
    Drone_Roll     int16
    Drone_Yaw      int16
  Voxel[0..Voxel_Count)
    X float32
    Y float32
    Z float32

エンディアンはシステム全体でリトルエンディアンに固定する。
"""
import struct

HEADER_FMT = "<IHHH3h"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 16 bytes
VOXEL_FMT = "<3f"
VOXEL_SIZE = struct.calcsize(VOXEL_FMT)  # 12 bytes

MTU_SAFE_PAYLOAD = 1200  # 1パケットのボクセルデータ上限(バイト) ※ヘッダー含め約1.2KB
MAX_VOXELS_PER_PACKET = MTU_SAFE_PAYLOAD // VOXEL_SIZE  # 100


def angle_to_fixed(deg: float) -> int:
    """度単位の角度を int16 (0.01度単位の固定小数点)にエンコードする"""
    v = int(round(deg * 100))
    return max(-32768, min(32767, v))


def pack_voxel_frame(frame_id: int, pitch: float, roll: float, yaw: float, voxels):
    """
    voxels: [(x, y, z), ...] (メートル単位)
    戻り値: 送信用パケット(bytes)のリスト。Voxel数に応じて複数パケットに分割する。
    """
    total = len(voxels)
    total_packets = max(1, (total + MAX_VOXELS_PER_PACKET - 1) // MAX_VOXELS_PER_PACKET)

    p_fixed = angle_to_fixed(pitch)
    r_fixed = angle_to_fixed(roll)
    y_fixed = angle_to_fixed(yaw)

    packets = []
    for packet_id in range(total_packets):
        chunk = voxels[packet_id * MAX_VOXELS_PER_PACKET:(packet_id + 1) * MAX_VOXELS_PER_PACKET]
        header = struct.pack(
            HEADER_FMT, frame_id & 0xFFFFFFFF, packet_id, total_packets, len(chunk), p_fixed, r_fixed, y_fixed
        )
        body = b"".join(struct.pack(VOXEL_FMT, x, y, z) for x, y, z in chunk)
        packets.append(header + body)
    return packets


def unpack_voxel_packet(data: bytes):
    """Unity側(参考実装/テスト用)。1パケットをデコードする。"""
    header = data[:HEADER_SIZE]
    frame_id, packet_id, total_packets, voxel_count, p_fixed, r_fixed, y_fixed = struct.unpack(HEADER_FMT, header)

    voxels = []
    offset = HEADER_SIZE
    for _ in range(voxel_count):
        x, y, z = struct.unpack(VOXEL_FMT, data[offset:offset + VOXEL_SIZE])
        voxels.append((x, y, z))
        offset += VOXEL_SIZE

    return {
        "frame_id": frame_id,
        "packet_id": packet_id,
        "total_packets": total_packets,
        "pitch": p_fixed / 100.0,
        "roll": r_fixed / 100.0,
        "yaw": y_fixed / 100.0,
        "voxels": voxels,
    }
