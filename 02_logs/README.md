# 02_logs/

各サービスのログ出力先。`docker-compose.yml` で以下のようにマウントされる。

```
02_logs/tello-ingest/tello-ingest.log
02_logs/ai-inference/ai-inference.log
02_logs/spatial-processor/spatial-processor.log
```

10MBごとにローテーションし、直近5世代分を保持する(`common/logging_config.py`)。
標準出力にも同時に出力されるため、`docker compose logs -f <service>` でも確認できる。
