"""
各サービス共通のロギング設定。
標準出力(docker logs用) と /app/logs/<service>.log (ローテーション, 02_logsにマウント) の両方に出力する。
"""
import logging
import os
from logging.handlers import RotatingFileHandler

DEFAULT_LOG_DIR = "/app/logs"


def setup_logging(service_name: str, log_dir: str = DEFAULT_LOG_DIR, level=logging.INFO):
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    try:
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, f"{service_name}.log")
        file_handler = RotatingFileHandler(file_path, maxBytes=10 * 1024 * 1024, backupCount=5)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError:
        # ログディレクトリがマウントされていない環境(単体テスト等)ではファイル出力をスキップ
        root.warning("log directory %s is not writable - file logging disabled", log_dir)
