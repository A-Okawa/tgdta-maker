#!/bin/bash
set -u
cd "$(dirname "$0")" || exit 1
if ! command -v python3 >/dev/null 2>&1; then
  printf '%s\n' 'Python 3.12 をインストールしてから実行してください。'
  exit 1
fi
python3 setup_local.py "$@"
