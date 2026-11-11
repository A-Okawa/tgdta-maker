#!/bin/bash
bash "$(dirname "$0")/setup_linux.sh" "$@"
status=$?
if [ -t 0 ]; then
  read -r -p 'Enter キーでこのウィンドウを閉じます。'
fi
exit "$status"
