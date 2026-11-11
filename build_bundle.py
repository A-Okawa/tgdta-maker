"""Create a source distribution using an explicit allowlist, without local data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile


BUNDLE_FILES = (
    "app.py", "requirements.txt", "tgdta_io.py", "tgdta_plot.py", "project_io.py", "figure_style.py",
    "style_templates.py", "palettes.py", "rich_text.py", "i18n.py", "colors.csv",
    "fonts/LiberationSans-Regular.ttf", "fonts/LiberationSans-Bold.ttf",
    "fonts/LiberationSans-Italic.ttf", "fonts/LiberationSans-BoldItalic.ttf",
    "fonts/LiberationSerif-Regular.ttf", "fonts/LiberationSerif-Bold.ttf",
    "fonts/LiberationSerif-Italic.ttf", "fonts/LiberationSerif-BoldItalic.ttf", "fonts/OFL.txt",
    "desktop.py", "setup_local.py", "build_bundle.py", "README.md",
    "setup_mac.command", "launch_mac.command", "setup_linux.sh", "launch_linux.sh",
    "setup_windows.bat", "launch_windows.bat",
)
BUNDLE_ROOT = "TG-DTA-Maker"


def create_bundle_bytes(source_dir: Path | str | None = None, wheelhouse: Path | str | None = None) -> bytes:
    """Bundle only application sources and optionally already-downloaded wheels.

    In particular, do not discover files recursively: credentials, user data,
    virtual environments, and draft files must never enter the download.
    """
    source = Path(source_dir).resolve() if source_dir is not None else Path(__file__).resolve().parent
    entries = []
    for name in BUNDLE_FILES:
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"配布用ファイルがありません (シンボリックリンクは不可): {name}")
        entries.append((name, path))
    if wheelhouse is not None:
        wheels = Path(wheelhouse)
        if wheels.is_symlink() or not wheels.is_dir():
            raise ValueError("wheelhouse は通常のディレクトリを指定してください。")
        wheel_files = sorted(wheels.glob("*.whl"))
        if not wheel_files:
            raise ValueError("wheelhouse に .whl がありません。")
        for path in wheel_files:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"通常の wheel ファイルのみ使用できます: {path.name}")
            entries.append((f"wheelhouse/{path.name}", path))
    result = io.BytesIO()
    manifest = {"format_version": 1, "distribution": "source", "files_sha256": {}}
    with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in entries:
            data = path.read_bytes()
            info = zipfile.ZipInfo(f"{BUNDLE_ROOT}/{name}")
            info.create_system = 3
            info.external_attr = ((0o100755 if name.endswith((".command", ".sh")) else 0o100644) << 16)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
            manifest["files_sha256"][name] = hashlib.sha256(data).hexdigest()
        archive.writestr(f"{BUNDLE_ROOT}/BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    return result.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description="認証情報・ユーザーデータを含まない PC 用 ZIP を作成")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "dist" / "TG-DTA-Maker-PC.zip")
    parser.add_argument("--wheelhouse", type=Path, help="同じ OS・CPU・Python 用に取得した wheel ファイルの保存先")
    args = parser.parse_args()
    payload = create_bundle_bytes(wheelhouse=args.wheelhouse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"{args.output.resolve()} ({len(payload):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
