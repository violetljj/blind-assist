"""Paired v75 panel and FP32 host comparison. Raw samples are retained under artifacts.local."""
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from eval_empirical import HERE, IO, OUT, PANEL, dump, model_path, sha

RUNTIME="/data/local/tmp/ba_qairt_htp_r0"


def inputs(kind,tag,mode):
    row = {k:PANEL/tag/(k+".raw") for k in ["rgb","grav_rays"]}
    row["host_focal"] = PANEL.parent/"focal_inputs"/(tag+".raw")
    if kind=="omni":
        row.update(tof_grid_s4=IO/("tof_"+mode+".raw"),has_tof=IO/("ht_zero.raw" if mode=="zero" else "ht_one.raw"))
    return row


def run(kind,variety,bits):
    exe=shutil.which("adb.exe") or shutil.which("adb")
    serial=os.environ.get("M13_DEVICE_SERIAL")
    if not serial:
        devices=subprocess.check_output([exe,"devices"],text=True,timeout=30)
        online=[s.split()[0] for s in devices.splitlines() if s.endswith("\tdevice")]
        if len(online)!=1:
            raise ValueError("Set M13_DEVICE_SERIAL for the intended S24 Ultra")
        serial=online[0]
    def adb(*argv):
        for attempt in range(4):
            p=subprocess.run([exe,"-s",serial,*map(str,argv)],capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=240)
            diagnostic=p.stdout+p.stderr
            if not p.returncode or not any(x in diagnostic for x in ("cannot connect to daemon","failed to start daemon","cannot connect to 127.0.0.1")):
                break
            time.sleep(1+attempt)
        if p.returncode:
            raise RuntimeError((p.stderr+p.stdout)[-3000:])
        return p.stdout.strip()
    def shell(*argv):
        return adb("shell",shlex.join(list(map(str,argv))))
    name=variety+"_"+kind+"_a"+str(bits)
    if variety=="fp16":
        context=IO/f"egogeom_omni_v41_{kind}_v75_fp16.bin"
        graph="v41_"+kind+"_final"
    else:
        folder=OUT/name
        b=json.loads((folder/"build.json").read_text())
        context=folder/"graph.bin"
        graph=b["graph_name"]
    model=shell("getprop","ro.product.model")
    assert model=="SM-S9280",model
    local=OUT/"device"/name
    local.mkdir(parents=True,exist_ok=True)
    if (local/"receipt.json").exists():
        prior=json.loads((local/"receipt.json").read_text())
        if not all(any(r["phase"]=="panel" and r["mode"]==m for r in prior["runs"])
                   for m in (["rgb"] if kind=="pure_rgb" else ["signed","zero"])):
            dump(local/("interrupted_"+str(time.time_ns())+".json"),prior)
    temp=shell("mktemp","-d","/data/local/tmp/m13_XXXXXXXX")
    assert re.fullmatch(r"/data/local/tmp/m13_[A-Za-z0-9]{8}",temp),temp
    env=["env",f"LD_LIBRARY_PATH={RUNTIME}",f"ADSP_LIBRARY_PATH={RUNTIME};/vendor/lib/rfsa/adsp;/dsp"]
    cache={}
    def ship(path):
        if path not in cache:
            h=sha(path)
            remote=temp+"/"+h[:18]+path.suffix
            adb("push",path,remote)
            assert shell("sha256sum",remote).split()[0]==h
            cache[path]=(remote,h)
        return cache[path][0]
    result={"device":model,"serial":serial,"context_sha256":sha(context),"context_bytes":context.stat().st_size,
            "graph":graph,"runs":[]}
    try:
        result["runtime_version"]=shell(*env,RUNTIME+"/qnn-net-run","--version").splitlines()[-1]
        cfg=json.loads((IO/("htp_rgb.json" if kind=="pure_rgb" else "htp_omni.json")).read_text())
        cfg["graphs"][0]["graph_names"]=[graph]
        dump(local/"htp.json",cfg)
        dump(local/"ext.json",{"backend_extensions":{"shared_library_path":RUNTIME+"/libQnnHtpNetRunExtensions.so",
                                                    "config_file_path":ship(local/"htp.json")}})
        rows=json.loads((PANEL/"frames.json").read_text())
        for mode in (["rgb"] if kind=="pure_rgb" else ["signed","zero"]):
            for phase in (["speed","panel"] if mode!="zero" else ["panel"]):
                tag=mode+"_"+phase
                chosen=rows[:1] if phase=="speed" else rows
                n=50 if phase=="speed" else len(chosen)
                listing=local/(tag+".txt")
                listing.write_text("\n".join(" ".join(k+":="+ship(p) for k,p in inputs(kind,r["id"],mode).items()) for r in chosen)+"\n")
                dest=temp+"/"+tag
                log=shell(*env,RUNTIME+"/qnn-net-run","--backend",RUNTIME+"/libQnnHtp.so",
                          "--config_file",ship(local/"ext.json"),"--retrieve_context",ship(context),
                          "--input_list",ship(listing),"--output_dir",dest,"--num_inferences",str(n),
                          "--keep_num_outputs",str(len(chosen)),"--perf_profile","burst","--profiling_level","basic")
                (local/(tag+".log")).write_text(log,encoding="utf-8")
                profile=shell(*env,RUNTIME+"/qnn-profile-viewer","--input_log",dest+"/qnn-profiling-data.log")
                (local/(tag+"_profile.txt")).write_text(profile)
                record={"mode":mode,"phase":phase,"n":n}
                for key,pat in [("htp_us",r"Backend \(Accelerator \(execute\) time\):\s*([\d.]+) us"),
                                ("netrun_us",r"NetRun:\s*([\d.]+) us")]:
                    match=re.search(pat,profile.split("Execute Stats (Average):")[-1])
                    if match:
                        record[key]=float(match[1])
                if phase=="panel":
                    for i,r in enumerate(rows):
                        adb("pull",dest+f"/Result_{i}/depth_m.raw",local/(mode+"_"+r["id"]+".raw"))
                result["runs"].append(record)
                print(name,record,flush=True)
    finally:
        shell("rm","-r",temp)
        result["released"]=adb("shell","test ! -e "+temp+" && echo yes")=="yes"
        result["input_sha256"]={str(p):h for p,(_,h) in cache.items()}
        dump(local/"receipt.json",result)


def reference(kind,rows):
    meta=OUT/"device"/f"host_{kind}_receipt.json"
    modes=["rgb"] if kind=="pure_rgb" else ["signed","zero"]
    identity={"onnx_sha256":sha(model_path(kind)),"inputs":{mode:{r["id"]:{k:sha(p) for k,p in inputs(kind,r["id"],mode).items()}
                   for r in rows} for mode in modes}}
    if meta.exists():
        record=json.loads(meta.read_text())
        if record["identity"]==identity and all((OUT/"device"/f"host_{kind}_{mode}_{r['id']}.npy").exists() for r in rows for mode in modes):
            return record["p50_p95_ms"]
    opts=ort.SessionOptions()
    opts.intra_op_num_threads=1
    opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(model_path(kind)),opts,providers=["CPUExecutionProvider"])
    timing=[]
    for mode in (["rgb"] if kind=="pure_rgb" else ["signed","zero"]):
        for row in rows:
            feeds=inputs(kind,row["id"],mode)
            x={t.name:np.fromfile(feeds[t.name],"<f4").reshape(t.shape) for t in session.get_inputs()}
            t0=time.perf_counter_ns()
            z=session.run(["depth_m"],x)[0].ravel()
            timing.append((time.perf_counter_ns()-t0)/1e6)
            np.save(OUT/"device"/f"host_{kind}_{mode}_{row['id']}.npy",z)
    duration=np.percentile(timing,[50,95]).tolist()
    dump(meta,{"identity":identity,"p50_p95_ms":duration})
    return duration


def report():
    rows=json.loads((PANEL/"frames.json").read_text())
    report={"panel_sha256":sha(PANEL/"frames.json"),"calibration_sha256":sha(OUT/"calibration.json"),
            "code_sha256":{p.name:sha(p) for p in HERE.glob("*.py")},"fixed_denominator_frames":len(rows),
            "code_hash_scope":"final reproduction source snapshot; individual build commands, logs and binary hashes are preserved",
            "metric":"mean pixel |device depth / ONNX FP32 depth - 1|, %; all pixels per frame",
            "alert_metrics":"not applicable: no alert policy, truth or route distance in fidelity panel",
            "host_onnx":{},"builds":{},"device":{}}
    report["protocol"]={"panel_domains":["aria","coda","day","dark"],"frames_per_domain":4,
                        "calibration_rgb":"8 distinct non-panel ARKit visit IDs with alternating day/dark transform",
                        "calibration_tof":"synthetic, separate from evaluation",
                        "evaluation_tof":"fixed archived signed grid replayed with each RGB input, or all-zero",
                        "focal_arm":"post-hoc; adds two copies of calibration rows with .2 and 4 as physical host factor stress inputs",
                        "device_timing":"50 repeated inferences of one input; HTP average, burst, no energy measurement"}
    for kind in ("pure_rgb","omni"):
        eq=OUT/f"equal_{kind}.json"
        allocation=OUT/("mixed_"+kind)/"allocation.json"
        if eq.exists() and allocation.exists():
            report.setdefault("equalization",{})[kind]={"fp32":json.loads(eq.read_text()),
                "allocation":json.loads(allocation.read_text()),
                "overrides_sha256":sha(allocation.with_name("overrides.json"))}
    report["fisher_sha256"]=sha(OUT/"fisher.json")
    report["theory_sha256"]=sha(OUT/"theory_verification.json")
    for p in OUT.glob("*/build.json"):
        report["builds"][p.parent.name]=json.loads(p.read_text())
        log=p.parent/"compile.log"
        if log.is_file():
            counts={k:int(v) for k,v in re.findall(r"\b(spill_bytes|fill_bytes|write_total_bytes|read_total_bytes)=(\d+)",log.read_text(errors="replace"))}
            report["builds"][p.parent.name]["compiled_DDR_bytes"]=counts
        encfile=p.parent/"graph_encoding.json"
        if not encfile.exists() and p.parent.name.startswith("matched_"):
            encfile=OUT/p.parent.name.replace("matched_","native_",1)/"graph_encoding.json"
        if encfile.is_file():
            actual=json.loads(encfile.read_text())["activation_encodings"]
            report["builds"][p.parent.name]["internal_host_focal_encoding"]=actual.get("host_focal_converted")
    for p in OUT.glob("*/conversion.json"):
        report.setdefault("conversions",{})[p.parent.name]=json.loads(p.read_text())
    for kind in ("pure_rgb","omni"):
        override=OUT/("mixed_"+kind)/"overrides.json"
        enc=OUT/("mixed_"+kind+"_a8")/"graph_encoding.json"
        if override.exists() and enc.exists():
            selected=json.loads(override.read_text())["activation_encodings"]
            actual=json.loads(enc.read_text())["activation_encodings"]
            report.setdefault("mixed_override_audit",{})[kind]={k:actual.get(k,[{}])[0].get("bitwidth") for k in selected}
    for kind in ("pure_rgb","omni"):
        report["host_onnx"][kind]={"sha256":sha(model_path(kind)),"cpu_p50_p95_ms":reference(kind,rows)}
    for rec in sorted((OUT/"device").glob("*/receipt.json")):
        folder=rec.parent
        info=json.loads(rec.read_text())
        kind="omni" if "_omni_" in folder.name else "pure_rgb"
        metrics={}
        for mode in (["rgb"] if kind=="pure_rgb" else ["signed","zero"]):
            if not any(r["phase"]=="panel" and r["mode"]==mode for r in info["runs"]):
                continue
            group=[]
            for row in rows:
                tag=row["id"]
                ref=np.load(OUT/"device"/f"host_{kind}_{mode}_{tag}.npy")
                z=np.fromfile(folder/(mode+"_"+tag+".raw"),"<f4")
                assert z.shape==ref.shape==(288*384,) and np.isfinite(z).all() and np.all(ref>0)
                group.append(float(100*np.mean(np.abs(z.astype(float)/ref-1))))
            metrics[mode]={"by_frame_pct":group,"mean_pct":float(np.mean(group)),"n_frames":len(group)}
            metrics[mode]["by_domain_mean_pct"]={d:float(np.mean([v for r,v in zip(rows,group) if r["domain"]==d]))
                                                  for d in sorted({r["domain"] for r in rows})}
        if "zero" in metrics:
            rgb=folder.with_name(folder.name.replace("_omni_","_pure_rgb_"))
            if (rgb/"rgb_aria_0.raw").exists():
                zero=max(float(np.max(np.abs(np.fromfile(rgb/("rgb_"+r["id"]+".raw"),"<f4")
                                    -np.fromfile(folder/("zero_"+r["id"]+".raw"),"<f4")))) for r in rows)
                metrics["no_tof_max_abs_vs_pure_rgb_m"]=zero
        report["device"][folder.name]={**{k:info[k] for k in ["runs","context_sha256","context_bytes","released"]},"fidelity":metrics}
    dump(OUT/"v5_empirical_report.json",report)
    print(json.dumps({k:{m:v.get("mean_pct") if isinstance(v,dict) else v for m,v in d["fidelity"].items()}
                      for k,d in report["device"].items()},indent=2))
