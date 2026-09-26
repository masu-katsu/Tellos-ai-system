# NVIDIA公式のTensorRTイメージを使用(TensorRT/CUDA/cuDNNを含む)。
# RTX 3050(Ampere, sm_86)に対応したドライバ/CUDAバージョンに合わせてタグを調整すること。
FROM nvcr.io/nvidia/tensorrt:24.05-py3

WORKDIR /app

COPY ai-inference/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY common ./common
COPY ai-inference/src ./src

ENV PYTHONUNBUFFERED=1
WORKDIR /app/src

CMD ["python3", "main.py"]
