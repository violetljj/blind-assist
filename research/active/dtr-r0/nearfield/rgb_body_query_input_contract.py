"""Read-only ARKit lowres input/metric-chain audit on all six cached captures.

This checks source timestamp/bytes, native dimensions/K/mm conversion, cached
registration maps, preprocessing and existing inference-source provenance. It
does not establish physical RGB/LiDAR alignment or run any model/GPU code.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
from PIL import Image
from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_interval_distribution import load, utc
from rgb_body_query_negative_frozen_score import paths
from rgb_body_query_scene_diagnostic import write_csv

WEIGHT_SHA='3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce'
CAPTURES=('47333462','40777060','40777065','40777073','40809740','40958733')
ASSETS=(('lowres_wide','.png'),('lowres_depth','.png'),('lowres_wide_intrinsics','.pincam'))


def source_functions(path,names):
    text=Path(path).read_text('utf-8-sig');tree=ast.parse(text)
    found={node.name:ast.get_source_segment(text,node) for node in ast.walk(tree)
           if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names}
    return found


def run(repo,output,budget_s):
    started=time.perf_counter();work,_,candidate,folders=paths(repo)
    runroot=work/'rgb-body-query-input-numerics-dev-20261010'
    if not output.resolve().is_relative_to((runroot/'input-audit').resolve()):
        raise ValueError('Inputaudit payload must remain in own run input-audit/')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'plan.json').exists():raise FileExistsError('Preserve inputaudit attempt')
    inputs={};checks=Counter();trace=[];records=[];warnings=[]
    receipt=dict(status='INCOMPLETE',started_utc=utc(),budget_cpu_wall_s=budget_s,
        placement='TASK_NOT_GPU_SUITABLE: source/bytes/CPU preprocessing contract audit',gpu_s=0,inference_calls=0,training_calls=0,download_bytes=0,
        resources='No workers/services/GPU/model instances retained')
    def check():
        if time.perf_counter()-started>=budget_s:raise TimeoutError('Input audit CPU wall budget reached')
    def register(path,expected=None):
        check();p=Path(path);key=str(p)
        if key not in inputs:inputs[key]=dict(path=key,sha256=sha(p))
        if expected is not None and inputs[key]['sha256']!=expected:raise ValueError(f'Input hash mismatch: {p}')
        return p
    try:
        parent=load(register(runroot/'plan.json'))
        assert parent['run']=='RGB_BODY_QUERY_INPUT_NUMERICS_DEV_20261010' and budget_s<=parent['budgets']['input_audit_wall_s']
        upstream=work/'ba-nfo-depthpro-20260919/upstream/src'
        upath=register(upstream/'depth_pro/depth_pro.py')
        functions=source_functions(upath,{'create_model_and_transforms','infer'})
        infer=functions['infer'];transform=functions['create_model_and_transforms']
        assert '_, _, H, W = x.shape' in infer and 'inverse_depth = canonical_inverse_depth * (W / f_px)' in infer
        assert 'torch.clamp(inverse_depth, min=1e-4, max=1e4)' in infer
        assert 'if f_px is None:' in infer and 'size=(self.img_size, self.img_size)' in infer
        assert 'Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])' in transform
        assert transform.index('ToTensor()') < transform.index('Normalize(') < transform.index('ConvertImageDtype(precision)')
        trace.append(dict(kind='upstream_metric_contract',path=str(upath),sha256=sha(upath),functions=functions,
            interpretation='Input H/W recorded before internal1536 square resize; canonical inverse depth * original RGB W/provided fx; inverse upsample to original H/W then reciprocal clamp; suppliedfx overrides predictedFOV. No fy/principalpoint correction.' ))
        modelidpath=register(work/'rgb-body-query-arkit-cal-dev-20261010/additional-inference/model_identity.json')
        modelid=load(modelidpath)
        for r in modelid['upstream_python_files']:register(upstream/r['path'],r['sha256'])
        codefingerprint=hashlib.sha256(json.dumps(modelid['upstream_python_files'],sort_keys=True).encode()).hexdigest()
        assert codefingerprint==modelid['upstream_code_sha256']
        checks['newcal_upstream_python_files_hash_match']=len(modelid['upstream_python_files'])
        register(work/'ba-nfo-depthpro-20260919/depth_pro.pt',WEIGHT_SHA)
        calroot=work/'rgb-body-query-arkit-cal-dev-20261010'
        oldsource=work/'rgb-body-query-query-level-dev-20261009/third-camera-source/arkit-additional'
        pairs=[]
        oldprep=work/'rgb-body-query-transfer-dev-20261009/camera-sensor'
        oldreceipt=load(register(oldprep/'preparation_receipt.json'))
        snap=register(oldprep/'executed_arkit_prepare.py',oldreceipt['source_sha256'])
        trace.append(dict(kind='old47333462_preparation_snapshot',path=str(snap),sha256=sha(snap),expected_receipt_sha256=oldreceipt['source_sha256']))
        addreceipt=load(register(oldsource/'completion_receipt.json'))
        addsource=register(oldsource/'executed_arkit_additional.py',addreceipt['source_sha256'])
        trace.append(dict(kind='old40777060_65_preparation_snapshot',path=str(addsource),sha256=sha(addsource)))
        calreceipt=load(register(calroot/'source/terminal_receipt.json'))
        calsource=register(calroot/'source/executed_arkit_cal_reference.py',calreceipt['source_sha256'])
        trace.append(dict(kind='cal_preparation_snapshot',path=str(calsource),sha256=sha(calsource)))
        for capture in CAPTURES:
            if capture=='47333462':
                manifestpath=folders['arkit16']/'dataset_manifest.json';source=work/'rgb-body-query-transfer-dev-20261009/camera-source'
                predpath=oldprep/'depthpro/predictions.json';obspath=oldprep/'observations.json';headpath=candidate/'predictions/arkit16/predictions.json'
            elif capture in ('40777060','40777065'):
                folder=folders['arkit_'+capture]
                manifestpath=folder/'dataset_manifest.json';source=oldsource/capture
                predpath=folder/'depthpro/predictions.json';obspath=folder/'observations.json';headpath=candidate/'predictions'/('arkit_'+capture)/'predictions.json'
            else:
                manifestpath=calroot/'reference/dataset_manifest.json';source=calroot/'source'/capture
                predpath=calroot/'additional-inference/predictions.json';obspath=calroot/'additional-inference/observations.json';headpath=calroot/'head/predictions.json'
            manifest,pred,obs,head=[load(register(p)) for p in (manifestpath,predpath,obspath,headpath)]
            assert pred['status']=='COMPLETE' and pred['weight_sha256']==WEIGHT_SHA and pred['evaluator_inputs'] is False and pred['scale_fit'] is False
            assert pred['observations_sha256']==sha(obspath)
            refs=[r for r in manifest['rows'] if r['scan']=='arkitscenes_'+capture]
            assert len(refs)==16
            plook={(r['scan'],r['frame']):r for r in pred['rows']};hlook={(r['scan'],r['frame']):r for r in head['rows']};olook={(r['scan'],r['frame']):r for r in obs['rows']}
            if capture in ('40777073','40809740','40958733'):
                script=register(calroot/'additional-inference/runner_executed.py',pred['script_sha256'])
                assert pred['upstream_code_sha256']==codefingerprint
                snaptext=script.read_text('utf-8-sig')
                assert "rgb = image.convert('RGB')" in snaptext and "f_px=torch.tensor(row['color_K'][0][0], device='cuda')" in snaptext
                trace.append(dict(kind='cal_inference_snapshot',capture=capture,path=str(script),sha256=sha(script),upstream_code_sha256=codefingerprint))
            else:
                warnings.append(dict(capture=capture,scope='Historical inference runner/upstream snapshot',status='PARTIAL_PROVENANCE',reason='Old prediction receipt stores weight/backend/observation identity and suppliedfx intent, but no historical inference script or upstream Python hash; current baseline source separately traced, not asserted identical to historical execution'))
            acquisition=load(register(source/'download_receipt.json'))
            archive_expected={r['name']:r['sha256'] for r in acquisition.get('rows',[])}
            archives=[]
            for asset,suffix in ASSETS:
                apath=register(source/(asset+'.zip'),archive_expected.get(asset+'.zip'))
                with zipfile.ZipFile(apath) as archive:
                    names=[n for n in archive.namelist() if n.endswith(suffix)]
                idx={Path(n).stem:n for n in names};assert len(idx)==len(names)
                archives.append(dict(asset=asset,path=apath,index=idx))
            for r in refs:
                p=plook[r['scan'],r['frame']];h=hlook[r['scan'],r['frame']];o=olook[r['scan'],r['frame']]
                assert p['rgb_sha256']==h['rgb_sha256']==o['rgb_sha256']==r['rgb_sha256']
                assert o['color_K']==h['color_K']==r['color_K'] and o['depth_K']==h['depth_K']==r['depth_K']
                register(r['rgb_path'],r['rgb_sha256']);register(r['reference_path'],r['reference_sha256']);register(p['path'],p['sha256']);register(h['sampled_depth_path'],h['sampled_depth_sha256'])
                assert all(r['source_id'] in a['index'] for a in archives)
                pairs.append(dict(capture=capture,ref=r,pred=p,head=h,archives=archives))
        assert len(pairs)==96 and len({(r['ref']['scan'],r['ref']['frame']) for r in pairs})==96
        baselinepath=register(Path(__file__).parent/'rgb_body_query_3rscan.py')
        baselinefunctions=source_functions(baselinepath,{'baseline','optical_z','sample_prediction'})
        trace.append(dict(kind='current_old_inference_function_only',path=str(baselinepath),sha256=sha(baselinepath),functions=baselinefunctions,historical_match='UNPROVEN_NO_RUNNER_SHA'))
        write(output/'inputs.json',dict(status='REGISTERED_BEFORE_FRAME_ANALYSIS',records=list(inputs.values()),frames=96,captures=CAPTURES))
        write(output/'plan.json',dict(created_utc=utc(),parent_plan_sha256=sha(runroot/'plan.json'),source_sha256=sha(__file__),frames=96,captures=CAPTURES,budget_cpu_wall_s=budget_s,new_inference=False,downloads=0,
            limits='Allcached6ARKitcaptures16frames each, no additions; native data contract does not prove physical alignment'))
        # Same upstream tensor sequence on CPU; no model import/construction.
        import torch
        torch.set_num_threads(2)
        from torchvision.transforms import Compose,ToTensor,Normalize,ConvertImageDtype
        transform_cpu=Compose([ToTensor(),Normalize([.5,.5,.5],[.5,.5,.5]),ConvertImageDtype(torch.float16)])
        opened={}
        try:
            for item in pairs:
                check();r,p,h=item['ref'],item['pred'],item['head'];name=r['source_id']
                payloads={};membernames={}
                for a in item['archives']:
                    if a['path'] not in opened:opened[a['path']]=zipfile.ZipFile(a['path'])
                    member=a['index'][name];payloads[a['asset']]=opened[a['path']].read(member);membernames[a['asset']]=member
                assert len({Path(n).stem for n in membernames.values()})==1
                timestamp=float(name.rsplit('_',1)[1]);assert timestamp==r['timestamp_s']
                for asset,key in [('lowres_wide','source_rgb_sha256'),('lowres_depth','source_depth_sha256'),('lowres_wide_intrinsics','calibration_sha256')]:
                    assert hashlib.sha256(payloads[asset]).hexdigest()==r[key]
                rgb_source=Image.open(io.BytesIO(payloads['lowres_wide']));rgb=rgb_source.convert('RGB')
                with Image.open(r['rgb_path']) as cached_rgb:assert np.array_equal(np.asarray(cached_rgb.convert('RGB')),np.asarray(rgb))
                raw=np.asarray(Image.open(io.BytesIO(payloads['lowres_depth'])))
                w,hgt,fx,fy,cx,cy=np.fromstring(payloads['lowres_wide_intrinsics'].decode(),sep=' ')
                assert (w,hgt)==rgb.size==(256,192) and raw.shape==(192,256) and raw.dtype==np.uint16
                k=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],np.float64)
                assert np.isfinite(k).all() and min(fx,fy)>0 and r['depth_shift']==1000.
                assert r['color_shape']==r['depth_shape']==[192,256] and np.array_equal(k,r['color_K']) and np.array_equal(k,r['depth_K'])
                gt=raw.astype(np.float32)/1000.;gt[raw==0]=np.nan
                yy,xx=np.indices(raw.shape,dtype=np.float32)
                with np.load(r['reference_path'],allow_pickle=False) as f:
                    assert np.array_equal(f['depth'],gt,equal_nan=True)
                    assert np.array_equal(f['depth_K'],k) and np.array_equal(f['color_K'],k)
                    assert np.array_equal(f['map_x'],xx) and np.array_equal(f['map_y'],yy) and f['observed'].all()
                with np.load(p['path'],allow_pickle=False) as f:dp=f['depth'].copy()
                with np.load(item['head']['sampled_depth_path'],allow_pickle=False) as f:sampled=f['depth']
                assert dp.shape==raw.shape and dp.dtype==np.float32 and np.isfinite(dp).all() and (dp>0).all()
                assert np.array_equal(sampled,dp)
                arr=np.asarray(rgb);tensor=transform_cpu(rgb)
                expected=((arr.astype(np.float32)/255.-.5)/.5).transpose(2,0,1).astype(np.float16)
                assert tensor.device.type=='cpu' and tensor.dtype==torch.float16 and list(tensor.shape)==[3,192,256]
                assert np.array_equal(tensor.numpy(),expected) and np.isfinite(expected).all() and expected.min()>=-1 and expected.max()<=1
                tensorfx32=np.float32(fx);w_over_fx32=np.float32(256.)/tensorfx32
                checks['exact_source_timestamp_and_SHA_frames']+=1;checks['native_K_units_identity_registration_frames']+=1
                checks['CPU_RGB_preprocessing_exact_frames']+=1;checks['existing_prediction_same_native_sampling_frames']+=1
                records.append(dict(capture=item['capture'],scan=r['scan'],frame=r['frame'],source_id=name,timestamp_s=timestamp,
                    rgb_member=membernames['lowres_wide'],depth_member=membernames['lowres_depth'],pincam_member=membernames['lowres_wide_intrinsics'],
                    source_RGB_mode=rgb_source.mode,native_width=256,native_height=192,fx=fx,fy=fy,cx=cx,cy=cy,
                    native_to_pincam_scale_x=1.,native_to_pincam_scale_y=1.,fx_tensor_float32=float(tensorfx32),infer_native_W_over_fpx_float32=float(w_over_fx32),
                    principal_x_offset_pixels=cx-127.5,principal_y_offset_pixels=cy-95.5,
                    raw_mm_min=int(raw.min()),raw_mm_max=int(raw.max()),missing_depth_pixels=int((raw==0).sum()),
                    valid_GT_median_m=float(np.nanmedian(gt)),RGB_value_min=int(arr.min()),RGB_value_max=int(arr.max()),
                    RGB_channel_means=json.dumps(arr.mean((0,1)).tolist()),PNG_metadata_keys=json.dumps(sorted(rgb_source.info.keys())),
                    EXIF_orientation=rgb_source.getexif().get(274),preprocess_max_abs=0.,DP_native_min_m=float(dp.min()),DP_native_max_m=float(dp.max()),
                    cached_DP10000_pixel_count=int((dp==10000.).sum()),contract='PASS'))
        finally:
            for archive in opened.values():archive.close()
        grouped=[]
        for capture in CAPTURES:
            rr=[r for r in records if r['capture']==capture]
            grouped.append(dict(capture=capture,frames=len(rr),fx_min=min(r['fx'] for r in rr),fx_max=max(r['fx'] for r in rr),
                fx_fy_equal=all(r['fx']==r['fy'] for r in rr),w_over_fx_min=min(r['infer_native_W_over_fpx_float32'] for r in rr),w_over_fx_max=max(r['infer_native_W_over_fpx_float32'] for r in rr),
                missing_GT_pixels=sum(r['missing_depth_pixels'] for r in rr),DP10000_pixels=sum(r['cached_DP10000_pixel_count'] for r in rr),native_pixels=16*192*256))
        write_csv(output/'frame_contracts.csv',records);write_csv(output/'capture_summary.csv',grouped)
        write(output/'source_trace.json',dict(records=trace,warnings=warnings))
        write(output/'checks.json',dict(status='PASS',checks=dict(checks)))
        write(output/'results.json',dict(status='COMPLETE',captures=grouped,frames=96,
            verified='Source same timestamp/stem/byteSHA; same256x192/pincam, fx=fy/native scale1, colorK=depthK; uint16 axial mm/1000 unchanged inclzero->NaN; mapsidentity/observedtrue; RGB byte and CPUtensor transform exact; cachedDP/native sampling equal; provided publicfx metric infer source chain and newcal executed/upstreamhash match',
            metric_chain=dict(infer_W='Original pre1536 RGB width256',provided_focal='public colorfx converted float32, not1536-rescaledfocal',scaling='canonical_inverse_depth*(256/f_px)',endpoint='1/clamp(inverse_depth,1e-4,1e4);10000m endpoint is positive finite cachedoutput, not filledGT',precision='Upstream transform converts tofloat16 after ToTensor+Normalize, including cachednative FP16 paths; numeric effect delegated to separate pairedGPU run'),
            unresolved=['Contract checks cannot prove physical RGB/LiDAR alignment or timestamp sensor accuracy','Historical old48 inference-runner/upstream SHA missing; old receipts publicRGB/weight/backend identities only; new48 fullsnapshot hashes match','Publicfx supplied; upstream does not use fy,cx,cy or distortion/exposure/color-profile correction; no alternative calibration/FOV substitution performed','No unique cause for crosssource metric bias established; no training/newworkingpoint or physical safety conclusion'],
            warnings=warnings,checks=dict(checks)))
        receipt['status']='COMPLETE'
    except Exception as error:
        receipt['failure']=repr(error);write(output/'failure.json',dict(error=repr(error),completed_frames=len(records)));raise
    finally:
        shutil.copyfile(__file__,output/'executed_input_contract.py')
        receipt.update(elapsed_cpu_wall_s=time.perf_counter()-started,source_sha256=sha(__file__),checks=dict(checks))
        write(output/'terminal.json',receipt)
    print('COMPLETE',receipt['elapsed_cpu_wall_s'],dict(checks),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--budget-s',type=float,default=240.)
    args=parser.parse_args();run(args.repo.resolve(),args.output.resolve(),args.budget_s)
