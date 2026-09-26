"""
入力映像を、アスペクト比を保ったまま 420x420 にレターボックス変換する(仕様書4章)。
黒帯部分は valid_region の外側として扱い、後段のDepth計算/ボクセル化の対象から除外する。
"""
import cv2
import numpy as np

TARGET = 420


def letterbox(rgb: np.ndarray):
    """
    rgb: (H,W,3) uint8
    戻り値:
      canvas: (TARGET,TARGET,3) uint8
      valid_region: (x0,y0,x1,y1) 実画像が配置されている範囲(黒帯を除く)
    """
    h, w = rgb.shape[:2]
    scale = TARGET / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((TARGET, TARGET, 3), dtype=np.uint8)
    x0 = (TARGET - new_w) // 2
    y0 = (TARGET - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    valid_region = (x0, y0, x0 + new_w, y0 + new_h)
    return canvas, valid_region
