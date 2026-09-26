# 03_models/depth-anything-v2/

`ai-inference` コンテナが読み込む TensorRT エンジンファイルをここに置く。

デフォルトのパス(docker-compose.yml / ai-inference/src/main.py の `DEPTH_ENGINE_PATH`):

```
03_models/depth-anything-v2/depth_anything_v2_vits.engine
```

## 変換手順の概要(RTX 3050 実機上で実行すること)

1. Depth Anything V2 (ViT-Small) の公開重みを取得する
2. ONNX形式にエクスポートする(入力: 1x3x420x420, 出力: 1x420x420 の相対深度)
3. `trtexec` でFP16の TensorRT エンジンに変換する

```bash
trtexec \
  --onnx=depth_anything_v2_vits.onnx \
  --saveEngine=depth_anything_v2_vits.engine \
  --fp16 \
  --shapes=input:1x3x420x420
```

生成した `.engine` ファイルは GPU/TensorRT/CUDAバージョンに依存するため、
実際に推論を行うマシン(RTX 3050 + 本番のTensorRTバージョン)上で変換すること。
このファイルが存在しない場合、`ai-inference` は起動時にエラーログを出し、
疑似的なダミー深度(輝度ベース)にフォールバックして動作を継続する。
