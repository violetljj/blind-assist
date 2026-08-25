#!/bin/sh
set -eu

display_number=99
Xvfb ":${display_number}" -screen 0 640x480x24 +extension GLX +render -noreset -ac -nolisten tcp &
xvfb_pid=$!
trap 'kill "$xvfb_pid" 2>/dev/null || true' EXIT INT TERM
export DISPLAY=":${display_number}"

attempt=0
while ! python -c 'import Xlib.display; display = Xlib.display.Display(); display.close()' >/dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 50 ]; then
        echo "Xvfb did not become ready" >&2
        exit 1
    fi
    sleep 0.1
done

"$@"
