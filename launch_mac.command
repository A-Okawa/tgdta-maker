#!/bin/bash
bash "$(dirname "$0")/launch_linux.sh" "$@"
status=$?
if [ "$status" -ne 0 ] && [ -t 0 ]; then
  read -r -p 'Enter キーでこのウィンドウを閉じます。'
fi
exit "$status"
