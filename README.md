# AI_Tello

Ryze Telloのリアルタイム3D空間認識システム — Dockerバックエンド一式。
(Unity側の受信・描画・UIは別プロジェクトとして構築する。ここに含まれるのは
「Tello -> AI推論 -> 3D化 -> UDP送信」までのDocker処理基盤)

## 構成

```
AI_Tello/
├── 01_main/                  Dockerコンテナ本体
│   ├── docker-compose.yml
│   ├── .env.example          環境変数の例(コピーして .env として使う)
│   ├── common/                サービス間で共有するコード
│   │   ├── shared_frame.py    RGBフレーム用 共有メモリ(最新1フレームのみ保持)
│   │   ├── shared_depth.py    Depthマップ用 共有メモリ(同上)
│   │   ├── udp_protocol.py    Unity向けUDPバイナリのpack/unpack
│   │   └── logging_config.py  共通ロギング設定(02_logsへの出力)
│   ├── tello-ingest/          ① Tello通信・H.264デコード・420x420化・IMU/ToF取得
│   ├── ai-inference/          ② Depth Anything V2 (TensorRT/RTX 3050) による深度推定
│   └── spatial-processor/     ③ 絶対距離補正・3D座標変換・ボクセル化・UDP送信
├── 02_logs/                   各サービスのログ出力先
└── 03_models/
    └── depth-anything-v2/     TensorRTエンジンファイルの配置場所
```

## 起動方法

```bash
cd 01_main
cp .env.example .env   # 必要に応じて値を編集(TELLO_IP, UNITY_HOST/PORTなど)
docker compose up --build
```

前提:
- PCがTelloのWiFi(`TELLO-XXXXXX`)に接続されていること
- `ai-inference` はNVIDIA GPU(RTX 3050想定)を使うため、
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  が導入済みであること
- `03_models/depth-anything-v2/depth_anything_v2_vits.engine` にモデルを配置していること
  (未配置の場合は `ai-inference` がダミー深度で起動を継続する)

## 設計方針の要点

- **キューを持たない**: 各段は常に「最新1フレームのみ」を共有メモリで受け渡す。
  処理が追いつかない場合は古いフレームを破棄し、遅延の蓄積を防ぐ。
- **Redisは使わない**: 過去データの蓄積より低遅延を優先する設計のため。
- **疎結合**: `spatial-processor` から先に渡すのはVoxel座標とDrone Poseのみ。
  生映像やDepthマップそのものはUnityに送らない。
- **IPC共有**: `tello-ingest` の `ipc: shareable` を起点に、他サービスは
  `ipc: "service:tello-ingest"` でIPC namespaceを共有し、
  `multiprocessing.shared_memory` + `posix_ipc` セマフォで同期する。

## Unity側との接続

`spatial-processor` は `UNITY_HOST:UNITY_PORT` へ `common/udp_protocol.py` の
フォーマット(Header + Voxel配列、リトルエンディアン、1パケット約1.2KB以下)で
UDPバイナリを送信するだけの単純な送信元として実装されている。
Unity側の受信・パケット再構成・GPU Instancing描画は本リポジトリの範囲外。
