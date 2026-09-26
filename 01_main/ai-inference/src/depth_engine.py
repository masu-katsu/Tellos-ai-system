"""
Depth Anything V2 (ViT-Small, TensorRT FP16) の推論をラップする(仕様書7〜8章)。

RTX 3050 + TensorRT が無い環境(このリポジトリを編集しているPCなど)でも
import・起動できるよう、TensorRT/pycudaが無い場合は「輝度ベースの疑似深度」に
フォールバックする。実機ではモデルを .engine に変換して 03_models/depth-anything-v2/
に配置すること。
"""
import logging

import numpy as np

logger = logging.getLogger("depth_engine")

try:
    import pycuda.autoinit  # noqa: F401  (CUDAコンテキストの暗黙初期化)
    import pycuda.driver as cuda
    import tensorrt as trt

    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False
    logger.warning("TensorRT/pycuda が見つかりません。DUMMY depth モードで起動します。")


class DepthEngine:
    def __init__(self, engine_path: str, input_size: int = 420):
        self.input_size = input_size
        self.engine_path = engine_path
        self._trt_ready = False

        if _TRT_AVAILABLE:
            self._load_engine(engine_path)
        else:
            logger.warning("DepthEngine: TensorRTなしで起動(ダミー出力)")

    def _load_engine(self, engine_path: str):
        trt_logger = trt.Logger(trt.Logger.WARNING)
        try:
            with open(engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self._allocate_buffers()
            self._trt_ready = True
            logger.info("TensorRTエンジンを読み込みました: %s", engine_path)
        except FileNotFoundError:
            logger.error("エンジンファイルが見つかりません: %s (ダミーモードで継続)", engine_path)

    def _allocate_buffers(self):
        self.stream = cuda.Stream()
        self.bindings = []
        self.host_inputs, self.host_outputs = [], []
        self.device_inputs, self.device_outputs = [], []

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = int(np.prod(shape))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
            else:
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)

    def infer(self, rgb_420: np.ndarray) -> np.ndarray:
        """rgb_420: (420,420,3) uint8 -> depth: (420,420) float32 相対深度"""
        if not self._trt_ready:
            return self._dummy_depth(rgb_420)

        x = rgb_420.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = x.transpose(2, 0, 1).ravel()

        np.copyto(self.host_inputs[0], x)
        cuda.memcpy_htod_async(self.device_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.device_outputs[0], self.stream)
        self.stream.synchronize()

        return self.host_outputs[0].reshape(self.input_size, self.input_size).copy()

    @staticmethod
    def _dummy_depth(rgb_420: np.ndarray) -> np.ndarray:
        """TensorRT無し環境向けの簡易フォールバック(輝度を疑似深度として使う。実運用では使用しない)"""
        gray = rgb_420.astype(np.float32).mean(axis=2)
        depth = 1.0 - (gray / 255.0)
        return depth.astype(np.float32)
