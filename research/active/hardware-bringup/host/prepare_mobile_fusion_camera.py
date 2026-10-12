"""Export existing Atom calibration/remap for the Android engineering port. No fitting/evaluation."""
from pathlib import Path
import json
import hashlib
import cv2
import numpy as np
from atom_camera import AtomCamera, CALIBRATION, ROOT


def main():
    out = ROOT / 'artifacts.local/work/mobile-fusion-integration-20261012'
    out.mkdir(parents=True, exist_ok=True)
    camera = AtomCamera()
    # Export exact OpenCV fixed-point maps, transposed into upright pixel order.
    maps = camera.map1.transpose(1, 0, 2).reshape(-1, 2).astype('<i2')
    fractions = camera.map2.T.reshape(-1).astype('<u2')
    packed = np.empty((len(maps), 3), dtype='<i2')
    packed[:, :2] = maps
    packed[:, 2] = fractions
    (out / 'atom_remap.bin').write_bytes(packed.tobytes())
    ext_path = CALIBRATION.parent / 'camera_tof_extrinsic.json'
    imu_path = ROOT / 'artifacts.local/hardware-bringup/atom-imu-v1/imu-mounting-operator-pilot.json'
    ext, imu = [json.loads(p.read_text(encoding='utf-8-sig')) for p in (ext_path, imu_path)]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    config = dict(schema='mobile-fusion-camera-v1', width=480, height=640, raw_width=640, raw_height=480,
                  K=camera.K_upright.tolist(), camera_from_tof=ext['rotation'],
                  tof_from_imu=imu['tof_from_imu_rotation'], atom_device_id=imu['atom_device_id'],
                  mounting_id=imu['mounting_id'], remap_sha256=sha(out / 'atom_remap.bin'),
                  sources={str(p.relative_to(ROOT)): sha(p) for p in (CALIBRATION, ext_path, imu_path)})
    (out / 'camera.json').write_text(json.dumps(config, indent=2)+'\n', encoding='utf8')
    # Deterministic pixel-field check covers edge/border and transpose conventions.
    raw = np.arange(480*640*3, dtype=np.uint32).reshape(480,640,3).astype(np.uint8)
    expected = camera.undistort_upright(raw)
    (out / 'remap_input.rgb').write_bytes(raw.tobytes())
    (out / 'remap_expected.rgb').write_bytes(expected.tobytes())
    print(json.dumps(dict(output=str(out), map_bytes=packed.nbytes, opencv=cv2.__version__)))


if __name__ == '__main__':
    main()
