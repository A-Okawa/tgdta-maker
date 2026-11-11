#!/bin/bash
set -u
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  printf '%s\n' '初回セットアップが必要です。setup ファイルを先に実行してください。'
  exit 1
fi
exec .venv/bin/python desktop.py "$@"
