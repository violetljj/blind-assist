"""Provision a task-owned ESP32 over USB without logging network credentials."""
import argparse
import json
import time
from pathlib import Path

import serial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--credentials", type=Path, required=True,
                        help="Ignored local JSON with ssid and password; never command-line secrets")
    parser.add_argument("--usb-otg", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.credentials.read_text(encoding="utf-8-sig"))
    ssid, password = config["ssid"], config["password"]
    if (not 1 <= len(ssid.encode("utf-8")) <= 32 or
            not 8 <= len(password.encode("utf-8")) <= 63 or
            any(c in ssid + password for c in "\t\r\n\x00")):
        raise SystemExit("Invalid credential format (values suppressed)")
    with serial.Serial(port=None, baudrate=115200, timeout=0.25, write_timeout=3) as link:
        link.dtr = args.usb_otg
        link.rts = False
        link.port = args.port
        link.open()
        time.sleep(1)
        link.reset_input_buffer()
        link.write(("WIFI\t" + ssid + "\t" + password + "\n").encode("utf-8"))
        link.flush()
        saved = False
        deadline = time.monotonic() + 35
        while time.monotonic() < deadline:
            # Parse only known machine events; never print arbitrary serial data.
            line = link.readline()
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError):
                continue
            if event.get("type") == "wifi_config_saved":
                saved = True
                print(json.dumps({"configured": True, "port": args.port}), flush=True)
            elif event.get("type") == "network_ready" and saved:
                print(json.dumps({"connected": True, "role": event.get("role"),
                                  "ip": event.get("ip")}), flush=True)
                return
        raise SystemExit("Credentials saved, connection unconfirmed" if saved else "No configuration receipt")


if __name__ == "__main__":
    main()
