#!/usr/bin/env python3
"""Seven fixed ESC-50 dominant-label vocabulary rejection comparisons.

CPU-only, no model inference. Fit folds 1/2, select thresholds on fold 3,
evaluate folds 4/5 once. Unknown means dominant label outside declared ten
classes, not actual sound absence or novelty to CLAP pretraining.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.dont_write_bytecode = True

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

KNOWN = ("dog", "crying_baby", "footsteps", "door_wood_knock", "clock_alarm",
         "glass_breaking", "car_horn", "siren", "engine", "toilet_flush")
FEATURES = ("whole_known_max_cosine", "whole_known_top1_top2_margin",
            "whole_winner_prompt_std", "whole_winner_window_mean",
            "whole_winner_window_std", "window_winner_agreement_fraction",
            "whole_winner_max_window_minus_whole")
METHODS = ("01_whole_no_rejection", "02_whole_max_cosine",
           "03_whole_margin", "04_window_classwise_max",
           "05_whole_prompt_ensemble", "06_logistic_global",
           "07_logistic_global_local")


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False)+"\n", encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def write_jsonl(path, rows):
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                              allow_nan=False)+"\n" for row in rows)
    Path(path).write_text(text, encoding="utf-8")
    return hash_file(path)


def rate(numerator, denominator):
    return {"n": int(numerator), "d": int(denominator),
            "value": float(numerator/denominator) if denominator else None}


def threshold_for_recall(scores, known, required=9):
    """Highest >= threshold accepting required known rows; ties accepted.

    No unknown score/label influences selection. Caller passes only calibration.
    """
    known_scores = sorted((float(s) for s,k in zip(scores,known) if k), reverse=True)
    if required <= 0 or required > len(known_scores):
        raise ValueError("Invalid required calibration recall count")
    if not all(math.isfinite(x) for x in known_scores):
        raise ValueError("Nonfinite calibration score")
    return known_scores[required-1]


def score_metrics(known, target, predicted, accepted, scores):
    known, target = np.asarray(known, bool), np.asarray(target)
    predicted, accepted = np.asarray(predicted), np.asarray(accepted, bool)
    scores = np.asarray(scores, dtype=float)
    if not (len(known)==len(target)==len(predicted)==len(accepted)==len(scores)):
        raise ValueError("Metric array lengths disagree")
    nk, nu = int(known.sum()), int((~known).sum())
    correct = predicted == target
    return {
        "rows":len(known), "accepted_known":rate((accepted & known).sum(),nk),
        "correct_and_accepted_known":rate((accepted & known & correct).sum(),nk),
        "accepted_unknown_primary_label_proxy":rate((accepted & ~known).sum(),nu),
        "known_top1_accuracy_before_rejection":rate((known & correct).sum(),nk),
        "known_classification_among_accepted":rate((known & accepted & correct).sum(),(known & accepted).sum()),
        "dominant_label_open_set_joint_correct_proxy":rate((known & accepted & correct).sum()+(~known & ~accepted).sum(),len(known)),
        "known_positive_auroc":float(roc_auc_score(known,scores)) if nk and nu else None,
        "known_positive_average_precision":float(average_precision_score(known,scores)) if nk and nu else None,
    }


def paired_counts(known, target, reference_pred, reference_accept, trial_pred, trial_accept):
    known, target = np.asarray(known,bool),np.asarray(target)
    ra,ta = np.asarray(reference_accept,bool),np.asarray(trial_accept,bool)
    rc = (np.asarray(reference_pred)==target) & ra
    tc = (np.asarray(trial_pred)==target) & ta
    return {
        "known_denominator":int(known.sum()), "unknown_primary_denominator":int((~known).sum()),
        "known_correct_accept_saved":int((known & tc & ~rc).sum()),
        "known_correct_accept_harmed":int((known & rc & ~tc).sum()),
        "known_accept_restored":int((known & ta & ~ra).sum()),
        "known_accept_removed":int((known & ra & ~ta).sum()),
        "unknown_primary_accept_removed_proxy":int((~known & ra & ~ta).sum()),
        "unknown_primary_accept_added_proxy":int((~known & ta & ~ra).sum()),
    }


def observable_features(whole, windows, known_ids):
    """Allowlisted observable scores only; does not accept evaluation labels."""
    raw = np.asarray([whole["cosine_scores"][c] for c in known_ids],dtype=float)
    ens = np.asarray([whole["ensemble_cosine_scores"][c] for c in known_ids],dtype=float)
    win = np.asarray([[row["cosine_scores"][c] for c in known_ids] for row in windows],dtype=float)
    winner = int(np.argmax(raw))  # known_ids numeric sorted, lower-id tie rule
    descending = np.sort(raw)[::-1]
    prompt_scores = np.asarray(whole["prompt_cosine_scores"][known_ids[winner]],dtype=float)
    if prompt_scores.shape != (3,) or win.shape != (3,10):
        raise ValueError("Expected exactly three prompts and three ten-class windows")
    winner_window = win[:,winner]
    features = np.asarray([descending[0], descending[0]-descending[1],
        prompt_scores.std(ddof=0), winner_window.mean(), winner_window.std(ddof=0),
        np.mean(np.argmax(win,axis=1)==winner), winner_window.max()-raw[winner]])
    if not np.isfinite(features).all() or not np.isfinite(ens).all():
        raise ValueError("Nonfinite observable score")
    return features, known_ids[winner], known_ids[int(np.argmax(win.max(axis=0)))], known_ids[int(np.argmax(ens))], float(win.max()), float(ens.max())


def self_check():
    # Hand example 1: the third-largest known score is .8, including its tie.
    threshold = threshold_for_recall([.9,.8,.8,.7,.99],[True,True,True,True,False],3)
    assert threshold == .8
    assert sum(s>=threshold for s in [.9,.8,.8,.7]) == 3
    assert threshold_for_recall([.9,.8,.8,.7,-99],[1,1,1,1,0],3)==threshold
    # Hand example 2: known accepts 2/3, correct accepted 1/3, unknown accept 1/2.
    m=score_metrics([1,1,1,0,0],[0,1,2,3,4],[0,2,2,0,0],[1,1,0,1,0],[.9,.8,.2,.7,.1])
    assert m["accepted_known"]==rate(2,3)
    assert m["correct_and_accepted_known"]==rate(1,3)
    assert m["accepted_unknown_primary_label_proxy"]==rate(1,2)
    assert m["known_top1_accuracy_before_rejection"]==rate(2,3)
    # Hand example 3: tied known/unknown scores have AUROC=.5 and AP=.5.
    m=score_metrics([1,0],[0,1],[0,0],[1,1],[.5,.5])
    assert m["known_positive_auroc"]==.5 and m["known_positive_average_precision"]==.5
    p=paired_counts([1,1,0],[0,1,3],[0,1,0],[1,0,1],[0,1,0],[0,1,0])
    assert p["known_correct_accept_saved"]==1 and p["known_correct_accept_harmed"]==1
    assert p["unknown_primary_accept_removed_proxy"]==1
    return {"status":"PASS","checks":["threshold highest allowed and ties; unknown not used",
        "hand known/unknown and correct accepted denominators", "tied AUROC/AP and paired rescue/harm"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path)
    parser.add_argument("--self-check",action="store_true")
    args=parser.parse_args()
    check=self_check()
    if args.self_check and args.root is None:
        print(json.dumps(check)); return
    if args.root is None:
        parser.error("--root required for real data")
    started=time.monotonic()
    root=args.root.resolve()
    out=root/"analysis"
    out.mkdir(exist_ok=True)
    if (out/"sealed_models_thresholds.json").exists() or (out/"summary.json").exists():
        raise FileExistsError("Completed/sealed analysis already exists; do not rerun or tune evaluation")
    plan=Path(__file__).with_name("PLAN.md")
    manifest=root/"data/evaluation.jsonl"
    whole_path=root/"inference/predictions.jsonl"
    window_path=root/"inference/window_predictions.jsonl"
    prompt_path=root/"class_prompts.json"
    provenance={"plan_sha256":hash_file(plan),"script_sha256":hash_file(__file__),
        "evaluation_manifest_sha256":hash_file(manifest),"whole_predictions_sha256":hash_file(whole_path),
        "window_predictions_sha256":hash_file(window_path),"class_prompts_sha256":hash_file(prompt_path)}
    rows=sorted(read_jsonl(manifest),key=lambda r:r["sample_id"])
    if len(rows)!=120 or len({r["sample_id"] for r in rows})!=120:
        raise ValueError("Expected exactly 120 unique manifest rows")
    prompts=json.loads(prompt_path.read_text(encoding="utf-8"))
    # Prompt labels use spaces, official ESC-50 categories use underscores.
    label_ids={r["label"].replace(" ","_"):str(r["class_id"]) for r in prompts["classes"]}
    known_ids=sorted((label_ids[label] for label in KNOWN),key=int)
    def phase(row):
        return "fit" if int(row["fold"]) in (1,2) else "calibration" if int(row["fold"])==3 else "evaluation"
    subsets={name:[i for i,r in enumerate(rows) if phase(r)==name] for name in ("fit","calibration","evaluation")}
    if {n:len(i) for n,i in subsets.items()}!={"fit":40,"calibration":20,"evaluation":60}:
        raise ValueError("Fixed fold counts differ")
    if Counter(int(r["fold"]) for r in rows)!=Counter({1:20,2:20,3:20,4:40,5:20}):
        raise ValueError("Fixed fold histogram differs")
    for row in rows:
        if not isinstance(row["known_primary_label"],bool) or row["known_primary_label"] != (row["label"] in KNOWN):
            raise ValueError("Known dominant-label flag mismatch")
        if str(row["target"])!=label_ids[row["label"]]:
            raise ValueError("Manifest class id/label mismatch")
        if (int(row["fold"])<=3)!=(row["role"]=="calibration"):
            raise ValueError("Manifest role/fold mismatch")
    split_contract={}
    for name,indices in subsets.items():
        subrows=[rows[i] for i in indices]
        nk=sum(r["known_primary_label"] for r in subrows)
        if nk*2!=len(subrows):
            raise ValueError("Expected known/unknown balanced partition")
        split_contract[name]={"rows":len(indices),"known":nk,"unknown_primary":len(subrows)-nk,
            "sources":sorted({str(r["source_id"]) for r in subrows}),
            "manifest_sha256":write_jsonl(out/(name+"_manifest.jsonl"),subrows)}
    for a,b in (("fit","calibration"),("fit","evaluation"),("calibration","evaluation")):
        for field in ("sample_id","source_id","audio_sha256"):
            if {str(rows[i][field]) for i in subsets[a]} & {str(rows[i][field]) for i in subsets[b]}:
                raise ValueError(f"{a}/{b} cross-partition {field} leak")
    dev_unknown={r["label"] for r in rows if phase(r)!="evaluation" and not r["known_primary_label"]}
    eval_unknown={r["label"] for r in rows if phase(r)=="evaluation" and not r["known_primary_label"]}
    if dev_unknown & eval_unknown:
        raise ValueError("Development/evaluation unknown dominant classes overlap")
    whole_list=read_jsonl(whole_path)
    whole={r["sample_id"]:r for r in whole_list}
    windows={}
    for row in read_jsonl(window_path):
        key=(row["sample_id"],int(row["window_idx"]))
        if key in windows: raise ValueError("Duplicate window inference row")
        windows[key]=row
    if len(whole_list)!=120 or set(whole)!={r["sample_id"] for r in rows} or len(windows)!=360:
        raise ValueError("Incomplete/extra/duplicate inference rows")
    feature_rows=[]
    prediction_ids={name:[] for name in METHODS}
    scores={name:[] for name in METHODS}
    for row in rows:
        sid=row["sample_id"]
        observation=whole[sid]
        views=[windows[(sid,j)] for j in range(3)]
        for obs in (observation,*views):
            if str(obs["source_id"])!=str(row["source_id"]) or obs["audio_path"]!=row["audio_path"]:
                raise ValueError("Inference/manifest identity mismatch")
        # Explicit allowlist: no manifest truth enters feature computation.
        observable_keys=("cosine_scores","ensemble_cosine_scores","prompt_cosine_scores")
        features,pred,wpred,epred,wscore,escore=observable_features(
            {k:observation[k] for k in observable_keys},
            [{k:view[k] for k in observable_keys} for view in views],known_ids)
        feature_rows.append(features)
        for name in METHODS:
            prediction_ids[name].append(wpred if name==METHODS[3] else epred if name==METHODS[4] else pred)
        for name in METHODS[:2]: scores[name].append(float(features[0]))
        scores[METHODS[2]].append(float(features[1]))
        scores[METHODS[3]].append(wscore);scores[METHODS[4]].append(escore)
    X=np.asarray(feature_rows,dtype=float)
    fit_idx=np.asarray(subsets["fit"])
    cal_idx=np.asarray(subsets["calibration"])
    y_fit=np.asarray([rows[i]["known_primary_label"] for i in fit_idx],dtype=int)
    fit_records={}
    for name,nfeatures in ((METHODS[5],2),(METHODS[6],7)):
        scaler=StandardScaler().fit(X[fit_idx,:nfeatures])
        clf=LogisticRegression(C=1.,solver="lbfgs",max_iter=1000,random_state=20261015)
        clf.fit(scaler.transform(X[fit_idx,:nfeatures]),y_fit)
        # Labels from calibration/evaluation never enter these transforms.
        scores[name]=clf.predict_proba(scaler.transform(X[:,:nfeatures]))[:,1].tolist()
        fit_records[name]={"feature_names":list(FEATURES[:nfeatures]),"C":1.0,"solver":"lbfgs",
            "max_iter":1000,"random_state":20261015,"penalty":"L2",
            "fit_rows":40,"fit_known":int(y_fit.sum()),"iterations":clf.n_iter_.tolist(),
            "scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist(),
            "standardized_coefficients":clf.coef_.tolist(),"standardized_intercept":clf.intercept_.tolist()}
    known_cal=[rows[i]["known_primary_label"] for i in cal_idx]
    thresholds={METHODS[0]:None}
    for name in METHODS[1:]:
        thresholds[name]=threshold_for_recall([scores[name][i] for i in cal_idx],known_cal,9)
    # Seal every trained weight and calibration threshold before evaluation.
    sealed={"contract":"PLAN.md fixed seven methods; highest threshold retaining >=9/10 fold3 known; ties accepted",
        "known_classes":list(KNOWN),"known_class_ids":known_ids,"splits":split_contract,
        "thresholds":thresholds,"models":fit_records,"inputs":provenance,
        "labels_used_for_fit":"only folds1/2","labels_used_for_threshold":"only fold3 known flags",
        "evaluation_selection":"none; no method selected on evaluation"}
    write_json(out/"sealed_models_thresholds.json",sealed)
    sealed_sha=hash_file(out/"sealed_models_thresholds.json")
    accepted={name:np.ones(len(rows),dtype=bool) if thresholds[name] is None else
              np.asarray(scores[name])>=thresholds[name] for name in METHODS}
    metrics={}
    for phase_name,indices in subsets.items():
        known=[rows[i]["known_primary_label"] for i in indices]
        targets=[str(rows[i]["target"]) for i in indices]
        metrics[phase_name]={name:score_metrics(known,targets,[prediction_ids[name][i] for i in indices],
            accepted[name][indices],[scores[name][i] for i in indices]) for name in METHODS}
    eval_idx=subsets["evaluation"]
    known_eval=[rows[i]["known_primary_label"] for i in eval_idx]
    targets_eval=[str(rows[i]["target"]) for i in eval_idx]
    comparisons=[(METHODS[1],METHODS[5]),(METHODS[1],METHODS[6]),(METHODS[5],METHODS[6])]
    paired={}
    for reference,trial in comparisons:
        paired[trial+"_vs_"+reference]=paired_counts(known_eval,targets_eval,
            [prediction_ids[reference][i] for i in eval_idx],accepted[reference][eval_idx],
            [prediction_ids[trial][i] for i in eval_idx],accepted[trial][eval_idx])
    decisions=[]
    for i,row in enumerate(rows):
        for name in METHODS:
            decisions.append({"sample_id":row["sample_id"],"source_id":row["source_id"],
                "fold":row["fold"],"phase":phase(row),"method":name,
                "gate_score":float(scores[name][i]),"threshold":thresholds[name],"ties_accepted":True,
                "accepted":bool(accepted[name][i]),"predicted_known_class_id":prediction_ids[name][i],
                "predicted_known_label":next(label for label,cid in label_ids.items() if cid==prediction_ids[name][i]),
                "output_label":next(label for label,cid in label_ids.items() if cid==prediction_ids[name][i]) if accepted[name][i] else None,
                "known_primary_label":row["known_primary_label"],"gt_primary_label":row["label"],
                "correct_dominant_class_if_known":bool(prediction_ids[name][i]==str(row["target"])) if row["known_primary_label"] else None,
                "observable_gate_features":{key:float(val) for key,val in zip(FEATURES,X[i])},
                "sealed_models_thresholds_sha256":sealed_sha})
    decisions_sha=write_jsonl(out/"per_sample_decisions.jsonl",decisions)
    # Strictly separate fifty-class argmax diagnostic, with no rejection.
    diagnostic={}
    for phase_name,indices in {**subsets,"all":list(range(len(rows)))}.items():
        correct=0
        for i in indices:
            obs=whole[rows[i]["sample_id"]]
            class_ids=sorted((k for k in obs["cosine_scores"] if k.isdigit()),key=int)
            if len(class_ids)!=50: raise ValueError("Diagnostic expects fifty classes")
            predicted=max(class_ids,key=lambda cid:(float(obs["cosine_scores"][cid]),-int(cid)))
            correct+=predicted==str(rows[i]["target"])
        diagnostic[phase_name]=rate(correct,len(indices))
    report={"status":"COMPLETE","evidence_label":"Development, primary-label-outside-vocabulary proxy only",
        "truth_limit":"ESC-50 dominant class only; not exhaustively labeled event absence or real-world false alarms; unknown may have been in CLAP pretraining",
        "metrics":metrics,"evaluation_paired_counts":paired,"raw_50class_first_prompt_accuracy_diagnostic":diagnostic,
        "thresholds":thresholds,"split_validation":{"rows_and_sources_disjoint":True,"audio_hashes_disjoint":True,
            "unknown_dominant_classes_disjoint":True,"histogram":dict(Counter(int(r["fold"]) for r in rows))},
        "cost_differences":{"01_02_03":"one whole audio view, first prompt, no fitted model",
            "04":"three window audio views, first prompt; same window count as local evidence gate",
            "05":"one whole audio view, three precomputed text prompts; ensemble defined by inference runner",
            "06":"one whole audio view and first-prompt max/margin; fits40 and calibrates20",
            "07":"whole plus three window audio views and three whole-winner prompts; fits40 and calibrates20",
            "note":"All views already inferred; analysis CPU timing does not establish inference FPS or deployment latency"},
        "inputs":provenance,"split_manifest_sha256":{k:v["manifest_sha256"] for k,v in split_contract.items()},
        "sealed_models_thresholds_sha256":sealed_sha,"decisions_sha256":decisions_sha,
        "self_check":check,"analysis_wall_seconds":time.monotonic()-started,"gpu_seconds":0,
        "new_download_bytes":0,"confidence_interval":"not computed; only30 known and30 unknown eval clips; no population-coverage claim"}
    write_json(out/"summary.json",report)
    lines=["# Fixed seven-method audio Development screen","",
        "Dominant-label vocabulary rejection proxy only; not actual target absence or real-world false alarms.","",
        "Fit folds1/2:40; threshold fold3:20; evaluation folds4/5:60. All row/source/audio-hash boundaries checked.","",
        "| Method | Accepted known /30 | Correct and accepted known /30 | Accepted unknown primary /30 | Known top1 /30 | AUROC | AP |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for name in METHODS:
        m=metrics["evaluation"][name]
        cells=[str(m[k]["n"])+"/"+str(m[k]["d"]) for k in ("accepted_known","correct_and_accepted_known","accepted_unknown_primary_label_proxy","known_top1_accuracy_before_rejection")]
        lines.append("| "+" | ".join([name,*cells,f'{m["known_positive_auroc"]:.4f}',f'{m["known_positive_average_precision"]:.4f}'])+" |")
    lines.extend(["", "No threshold or model was selected from evaluation. Learned gates use 40 fitted rows; window evidence requires extra audio views.",
        "", "Full denominators, coefficients, thresholds, hashes and paired rescue/harm counts are in summary.json and sealed_models_thresholds.json.",
        "", "No mixtures, true event-absence verification, spatial inference or BLV user benefit were tested."])
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(out/"receipt.json",{"status":"COMPLETE","script_sha256":hash_file(__file__),
        "outputs":{p.name:hash_file(p) for p in out.iterdir() if p.is_file()},
        "analysis_wall_seconds":time.monotonic()-started,"gpu_seconds":0,"new_download_bytes":0})
    print(json.dumps({"evaluation":metrics["evaluation"],"paired":paired,"diagnostic50":diagnostic,
        "thresholds":thresholds,"analysis_wall_seconds":time.monotonic()-started},ensure_ascii=False))


if __name__=="__main__":
    main()
