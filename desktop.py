"""Launch the installed TG-DTA Maker locally; never installs or downloads packages."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser


APP_DIR = Path(__file__).resolve().parent
REQUIRED_MODULES = ("streamlit", "numpy", "pandas", "matplotlib", "scipy", "plotly")


def default_data_dir() -> Path:
    """Keep local projects outside the application, so upgrades cannot erase them."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "TG-DTA Maker"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "TG-DTA Maker"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "tgdta-maker"


def local_environment(data_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "TGDTA_MAKER_LOCAL": "1",
        "TGDTA_MAKER_DATA_DIR": str(data_dir.resolve()),
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_SERVER_HEADLESS": "true",
        "MPLBACKEND": "Agg",
    })
    return env


def server_command(port: int) -> list[str]:
    return [
        sys.executable, "-m", "streamlit", "run", str(APP_DIR / "app.py"),
        "--server.address=127.0.0.1", f"--server.port={port}",
        "--server.headless=true", "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false", "--global.developmentMode=false",
    ]


def available_port(requested: int = 0) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", requested))
        return listener.getsockname()[1]


def validate_installation(data_dir: Path) -> None:
    if not (APP_DIR / "app.py").is_file():
        raise RuntimeError("app.py がありません。ZIP をすべて展開してから起動してください。")
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "必要なパッケージがありません: " + ", ".join(missing)
            + "。初回セットアップを実行してください。"
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=data_dir):
        pass


def wait_until_ready(process: subprocess.Popen, port: int, timeout: float = 60) -> None:
    # Explicitly bypass proxy configuration: this request must stay on this PC.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"アプリが起動前に終了しました (code={process.returncode})。")
        try:
            with opener.open(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.2)
    raise RuntimeError("アプリの起動が 60 秒以内に完了しませんでした。")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def request_shutdown(_signum, _frame) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TG-DTA Maker をこの PC のみで起動します。")
    parser.add_argument("--port", type=int, default=0, help="待受ポート (省略時は空きポート)")
    parser.add_argument("--data-dir", type=Path, default=None, help="自動保存とログの保存先")
    parser.add_argument("--no-browser", action="store_true", help="ブラウザーを自動で開かない")
    parser.add_argument("--check", action="store_true", help="依存関係と保存先のみ確認")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port は 0〜65535 の範囲で指定してください。")
    data_dir = (args.data_dir or Path(os.environ.get("TGDTA_MAKER_DATA_DIR") or default_data_dir())).expanduser().resolve()
    log_path = data_dir / "launcher.log"
    process = None
    try:
        validate_installation(data_dir)
        if args.check:
            print(f"起動準備 OK\nPython: {sys.executable}\n保存先: {data_dir}")
            return 0
        port = available_port(args.port)
        url = f"http://127.0.0.1:{port}"
        print(f"TG-DTA Maker を起動しています…\n保存先: {data_dir}\nログ: {log_path}", flush=True)
        options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                server_command(port), cwd=APP_DIR, env=local_environment(data_dir),
                stdout=log, stderr=subprocess.STDOUT, **options,
            )
            wait_until_ready(process, port)
            print(f"起動しました: {url}\n終了するには、このウィンドウで Ctrl+C を押してください。", flush=True)
            if not args.no_browser:
                try:
                    if not webbrowser.open(url):
                        print("上の URL をブラウザーで開いてください。", flush=True)
                except webbrowser.Error:
                    print("上の URL をブラウザーで開いてください。", flush=True)
            returncode = process.wait()
            if returncode:
                raise RuntimeError(f"アプリが終了しました (code={returncode})。")
        return 0
    except KeyboardInterrupt:
        print("\nTG-DTA Maker を終了します。", flush=True)
        return 0
    except (OSError, RuntimeError) as exc:
        print(f"起動できませんでした: {exc}\nログ: {log_path}", file=sys.stderr)
        return 1
    finally:
        if process is not None:
            stop_server(process)


if __name__ == "__main__":
    # Closing the terminal or stopping a launcher must also stop its child server.
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    raise SystemExit(main())
