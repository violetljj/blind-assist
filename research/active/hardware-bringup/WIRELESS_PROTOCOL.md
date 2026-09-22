# Direct Wi-Fi demo protocol

Both boards join the phone's 2.4 GHz hotspot (or the same local Wi-Fi). The PC
is used for initial development and provisioning, not transport at runtime.
`atom_wifi` and `tof_wifi` are separate sketches; original USB sketches remain.

Both provide `GET /api/status` on port 80: `role` (`camera` or `tof`),
`device_id` (MAC), random per-boot `boot_id`, `firmware`, and `endpoint`.
Local UDP port 3334 accepts the exact 18-byte ASCII `BADEMO_DISCOVER_V1` and
replies to the sender with this same JSON. Endpoint addresses are DHCP leases;
rediscover after network changes. Discovery is not authentication.

The Atom serves `http://IP:81/stream`, multipart MJPEG with `Content-Length`,
`X-Sequence-Id` (boot ID), `X-Frame-Sequence`, `X-Capture-Timestamp-Us`,
`X-Jpeg-Ready-Timestamp-Us`, `X-Device-Send-Start-Timestamp-Us`.
Capture means the ESP camera framebuffer timestamp, not a claim of exposure
synchronization. It has no single-zone ToF; `X-Tof-Valid` is false.
VGA, JPEG quality 12, two buffers, latest grab and TCP_NODELAY are used.
Socket send timeout is one second; the receiver must discard stale data.

The XIAO serves `GET /api/tof` with `boot_id`, `seq`, `sampled_us`, `send_us`,
`rows=8`, `cols=8`, and 64-element integer arrays `distance_mm`,
`target_status`, `nb_target`. Until the first successful read it returns 503.
Sample time is read completion; no exposure or RGB synchronization is implied.
Existing 5 Hz ranging and all raw validity values are retained. Readers must
reject stale/frozen sequences even when HTTP succeeds. A stalled sensor can
still return its old sample, identifiable from timestamp and sequence.

Both implement the existing 3333/UDP timing exchange: little-endian 16-byte
request (`BAT0`, uint32 ID, int64 client start ns), 24-byte response (`BAT1`,
same ID, int64 device receive us, int64 device send us). Keep a separate
mapping per board and invalidate it on boot ID change. A timestamp mapping
does not synchronize sensor exposures. Report RTT/error bounds with latency.

Wi-Fi sleep is disabled. One bounded physical-USB provisioning command is
`WIFI<TAB>ssid<TAB>password<LF>`. Credentials are stored only in device NVS;
the firmware never echoes them. `host/provision_wifi.py` reads an ignored
private JSON file containing `ssid` and `password`, sends it and prints only
save/connection status. Use `--usb-otg` for Atom. Never put credentials in
CLI arguments, source, logs or published receipts. Full flash backups may
contain NVS credentials and must remain local ignored artifacts.

Before upload, `host/verify_wifi_backup.py` checks a full 8 MB backup against
the retained USB application's bytes and existing app0 capacity. Write only
the application at `0x10000`; do not erase the partition table or NVS.
These are development demo interfaces on a user-controlled local network;
they are not encrypted or authenticated application protocols.
