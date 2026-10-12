"""Provision a task-owned ESP32 over USB without logging network credentials."""
import argparse
import json
import time
from pathlib import Path

def provisioning_command(config, slot=None):
    """Validate private input, returning only the USB command; errors contain no values."""
    if not isinstance(config, dict):
        raise ValueError("Invalid credential format (values suppressed)")
    profiles = config.get("profiles")
    if profiles is not None:
        if slot is not None or not isinstance(profiles, list) or len(profiles) != 2:
            raise ValueError("Expected two profiles; --slot applies only to single-profile input")
    else:
        profiles = [config]
    fields = []
    for profile in profiles:
        if not isinstance(profile, dict):
            raise ValueError("Invalid credential format (values suppressed)")
        ssid, password = profile.get("ssid"), profile.get("password")
        if (not isinstance(ssid, str) or not isinstance(password, str) or
                not 1 <= len(ssid.encode("utf-8")) <= 32 or
                not 8 <= len(password.encode("utf-8")) <= 63 or
                any(c in ssid + password for c in "\t\r\n\x00")):
            raise ValueError("Invalid credential format (values suppressed)")
        fields.extend((ssid, password))
    if len(profiles) == 2:
        if fields[0] == fields[2]:
            raise ValueError("Profiles must use distinct network names (values suppressed)")
        prefix = "WIFI2\t"
    elif slot is not None:
        if slot not in (0, 1):
            raise ValueError("Profile slot must be 0 or 1")
        prefix = f"WIFI_SLOT\t{slot}\t"
    else:
        prefix = "WIFI\t"
    return (prefix + "\t".join(fields) + "\n").encode("utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    credential_source = parser.add_mutually_exclusive_group(required=True)
    credential_source.add_argument("--credentials", type=Path,
                                   help="Ignored JSON {ssid,password} or {profiles:[{ssid,password},{ssid,password}]}")
    credential_source.add_argument("--password-profile",
                                   help="Read an existing Windows WLAN profile privately in memory")
    parser.add_argument("--ssid", help="Network name for --password-profile; no password argument is accepted")
    parser.add_argument("--usb-otg", action="store_true")
    parser.add_argument("--slot", type=int, choices=(0, 1),
                        help="Update one remembered network; --slot 1 preserves the existing LAN")
    args = parser.parse_args(argv)
    try:
        if args.password_profile:
            if not args.ssid:
                raise ValueError("Network name is required")
            from m3_wifi_setup import saved_password
            config = dict(ssid=args.ssid, password=saved_password(args.password_profile))
        else:
            config = json.loads(args.credentials.read_text(encoding="utf-8-sig"))
        command = provisioning_command(config, args.slot)
        del config
    except (OSError, ValueError, UnicodeError):
        raise SystemExit("Invalid credential file or format (values suppressed)") from None
    import serial
    with serial.Serial(port=None, baudrate=115200, timeout=0.25, write_timeout=3) as link:
        link.dtr = args.usb_otg
        link.rts = False
        link.port = args.port
        link.open()
        time.sleep(1)
        link.reset_input_buffer()
        if link.write(command) != len(command):
            raise SystemExit("USB configuration command was not sent completely (values suppressed)")
        link.flush()
        del command
        saved = False
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            # Parse only known machine events; never print arbitrary serial data.
            line = link.readline()
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError):
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "wifi_config_saved":
                saved = True
                print(json.dumps({"configured": True, "port": args.port,
                                  "profile_count": event.get("profile_count"),
                                  "profile_index": event.get("profile_index")}), flush=True)
            elif event.get("type") == "wifi_config_rejected":
                raise SystemExit("Firmware rejected Wi-Fi profiles (values suppressed)")
            elif event.get("type") == "network_ready" and saved:
                print(json.dumps({"connected": True, "role": event.get("role"),
                                  "ip": event.get("ip"), "boot_id": event.get("boot_id"),
                                  "network": event.get("network")}), flush=True)
                return
        raise SystemExit("Credentials saved, connection unconfirmed" if saved else "No configuration receipt")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Transport exceptions may contain writes; do not expose their messages or traceback.
        raise SystemExit("USB provisioning failed (values suppressed)") from None
