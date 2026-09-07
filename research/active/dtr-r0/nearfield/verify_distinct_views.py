"""Acquisition checks and contact sheet, before model scoring; no case selection."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/"tools"))
import cv2
import numpy as np
from near_field import Camera,NearFieldEncoder
from rgb_replay import _decode
from diagnose_depth_loss import sha


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--capture",type=Path,required=True)
    args=p.parse_args()
    root=args.capture.resolve()
    target=root/"verification.json"
    if target.exists(): raise ValueError("Preserve original acquisition check")
    receipt=json.loads((root/"receipt.json").read_text(encoding="utf-8"))
    released=json.loads((root/"process-release.json").read_text(encoding="utf-8"))
    assert receipt["status"]=="PASS" and receipt["source_unchanged"] and released["released"]
    manifest=json.loads((root/"model/sensor_manifest.json").read_text(encoding="utf-8"))
    frames=manifest["frames"]
    assert len(frames)==18 and [r["sample_index"] for r in frames]==list(range(18))
    decoded=[_decode(root/"model",r,manifest["calibration"]) for r in frames]
    camera=decoded[0]["camera"]
    # Small analytic calibration check, CPU TASK_NOT_GPU_SUITABLE.
    v,u=np.mgrid[:camera.height,:camera.width]
    f=camera.width/(2*np.tan(np.radians(camera.hfov_deg/2)))
    ray_z=np.sin(np.radians(camera.pitch_deg))+np.cos(np.radians(camera.pitch_deg))*(-(v-(camera.height-1)/2)/f)
    floor_height=-decoded[0]["native"][320:350,300:340]*ray_z[320:350,300:340]
    optical_error=float(np.max(np.abs(floor_height-camera.camera_height_m)))
    assert optical_error<.02,"Clear center-floor optical-height mismatch"
    tiles=[]
    identities=[]
    for row,sample in zip(frames,decoded):
        tile=np.zeros((205,320,3),np.uint8)
        tile[:180]=cv2.resize(sample["bgr"],(320,180))
        cv2.putText(tile,row["case_name"],(5,196),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        tiles.append(tile)
        identities.append(dict(name=row["case_name"],rgb_sha256=sample["rgb_sha256"],native_sha256=sample["native_sha256"],
                               native_valid_fraction=float((sample["native"]>0).mean())))
    sheet=np.concatenate([np.concatenate(tiles[i:i+3],axis=1) for i in range(0,18,3)],axis=0)
    assert cv2.imwrite(str(root/"contact-sheet.jpg"),sheet)
    changes=[]
    for i in (1,3,5,7,9):
        changes.append(dict(near=frames[i]["case_name"],far=frames[i+1]["case_name"],
                            native_pixels_changed_over_3cm=int((np.abs(decoded[i]["native"]-decoded[i+1]["native"])>.03).sum())))
    assert all(r["native_pixels_changed_over_3cm"]>0 for r in changes)
    result=dict(status="PASS",frames=18,source_unchanged=True,processes_released=True,
                clear_floor_max_optical_height_error_m=optical_error,pair_native_changes=changes,
                identities=identities,code_sha256=sha(Path(__file__)),
                scope="Acquisition dimensions, optical height, distinct native geometry; not obstacle accuracy or image-quality certification")
    target.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))


if __name__=="__main__": main()
