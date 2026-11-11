"""One-time local setup. A supplied wheelhouse enables a fully offline install."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv


APP_DIR = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TG-DTA Maker の専用 Python 環境を準備します。")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="wheelhouse のみからインストール")
    mode.add_argument("--online", action="store_true", help="wheelhouse があってもオンラインで取得")
    parser.add_argument("--wheelhouse", type=Path, default=APP_DIR / "wheelhouse")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 10):
        print("Python 3.10 以降が必要です。Python 3.12 を推奨します。", file=sys.stderr)
        return 1
    wheelhouse = args.wheelhouse.expanduser().resolve()
    offline = args.offline or (not args.online and any(wheelhouse.glob("*.whl")))
    if offline and not any(wheelhouse.glob("*.whl")):
        print(f"wheelhouse に .whl がありません: {wheelhouse}", file=sys.stderr)
        return 1
    environment = APP_DIR / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    try:
        if not python.is_file():
            print("専用環境 .venv を作成しています…", flush=True)
            venv.EnvBuilder(with_pip=True).create(environment)
        command = [str(python), "-m", "pip", "--disable-pip-version-check", "install"]
        if offline:
            print(f"オフラインでインストールします: {wheelhouse}", flush=True)
            command += ["--no-index", "--find-links", str(wheelhouse), "--only-binary=:all:"]
        else:
            print("初回セットアップ: インターネットから必要なパッケージを取得します。", flush=True)
        command += ["-r", str(APP_DIR / "requirements.txt")]
        subprocess.run(command, cwd=APP_DIR, check=True)
        subprocess.run([str(python), str(APP_DIR / "desktop.py"), "--check"], cwd=APP_DIR, check=True)
        print("セットアップ完了。次回から launch ファイルを開いてください。", flush=True)
        return 0
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"セットアップに失敗しました: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
