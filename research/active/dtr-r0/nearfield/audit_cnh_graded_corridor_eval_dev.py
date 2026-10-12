"""Independent NumPy replay of saved corridor scalar models and alert accounting.

Reads consumed caches only; does not call production fit/predict/metrics helpers.
Checks the metadata partition and reconstructs all model normalization, cutoffs,
grades, timely/late events, costs, paired first times and validation ledger.
"""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT/'artifacts.local/work/cnh-counterfactual-dev-20261009'
OUT = ROOT/'artifacts.local/work/cnh-graded-corridor-dev-20261010'
SEEDS = (2026100955,2026100956,2026100957)
CHECKS = 0

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def eq(actual, expected, where):
    global CHECKS
    if isinstance(expected,dict):
        assert set(actual)==set(expected),(where,'keys')
        for key in expected: eq(actual[key],expected[key],where+'/'+str(key))
    elif isinstance(expected,list):
        assert len(actual)==len(expected),(where,'length')
        for i,(a,b) in enumerate(zip(actual,expected)): eq(a,b,where+'/'+str(i))
    else:
        assert actual==expected,(where,actual,expected)
        CHECKS += 1

def smooth(raw):
    raw = np.asarray(raw,dtype=np.float64)
    answer = np.empty_like(raw)
    for t in range(13):
        lo = max(t-4,0)
        weight = np.exp2(np.arange(t-lo+1))
        answer[...,t,:] = np.sum(raw[...,lo:t+1,:]*weight[:,None],axis=-2)/weight.sum()
    return answer

def load():
    datasets = {}
    for spec in read(SRC/'eval_manifest.json')['datasets']:
        if spec['branch']!='ideal': continue
        rows = read(SRC/spec['scene_rows'])
        if isinstance(rows,dict):
            rows = rows.get(spec.get('rows_key'),rows.get('scene_rows',rows.get(spec['split'],rows.get('calibration'))))
        with np.load(SRC/spec['physics'],allow_pickle=False) as a:
            cat = a['category'].copy()
            ids = np.array([r.get('scene_id',r.get('id')) for r in rows])
            if 'scene_ids' in a: np.testing.assert_array_equal(a['scene_ids'],ids)
        with np.load(SRC/spec['baseline'],allow_pickle=False) as a:
            m3,local = smooth(a['m3_raw']),smooth(a['local_raw'])
        raw = {}
        for job in spec['scores']:
            if job['arm']!='ordinary': continue
            with np.load(SRC/job['path'],allow_pickle=False) as a:
                eq(a['seed'].item(),job['seed'],'raw/seed')
                eq(a['arm'].item(),'ordinary','raw/arm')
                eq(a['branch'].item(),'ideal','raw/branch')
                eq(a['split'].item(),spec['split'],'raw/split')
                raw[job['seed']] = a['raw'].copy()
        groups = {}
        for field,prefix in (('shape_family','family'),('background_family','background_family'),('geometry_family','geometry_family')):
            if all(field in r for r in rows):
                values = np.array([str(r[field]) for r in rows])
                groups.update({prefix+':'+v:values==v for v in set(values)})
        datasets[spec['split']] = dict(rows=rows,ids=ids,category=cat,m3=m3,local=local,raw=raw,groups=groups)
    return datasets

def score_features(raw,theta):
    slope = np.zeros_like(raw,dtype=float)
    noise = slope.copy()
    for t in range(13):
        y = raw[...,max(0,t-4):t+1,:]
        x = np.arange(y.shape[-2],dtype=float)
        x -= x.mean()
        if len(x)>1:
            # Preserve raw FP32 mean semantics; the least-squares slope is FP64.
            b = np.sum(y*x[:,None],axis=-2)/np.dot(x,x)
            slope[...,t,:] = b
            noise[...,t,:] = np.sqrt(np.mean((y-y.mean(-2)[...,None,:]-b[...,None,:]*x[:,None])**2,axis=-2))
    return np.stack((smooth(raw)-theta,slope,noise),axis=-1)

def spatial(split,contract):
    with np.load(OUT/'features'/f'{split}_features.npz',allow_pickle=False) as a:
        values,names,ids = a['features'],a['names'].tolist(),a['scene_ids']
        f = lambda name:values[...,names.index(name)].astype(float)
        primitives = contract['spatial_names'][:12]
        derived = [f('inner_current_positive_log_total')/np.maximum(f('inner_current_membership_mass'),1e-8),
            f('inner_past8_positive_log_total')/np.maximum(f('inner_past8_membership_mass'),1e-8),
            f('inner_current_top3_positive_peak_log_sum')/np.maximum(f('inner_current_positive_log_total'),1e-8),
            f('inner_current_top1_positive_peak_log')-f('inner_past8_top1_positive_peak_log'),
            f('inner_current_samebin_peak_log')-f('ring_current_samebin_peak_log')]
        x = np.stack([f(n) for n in primitives]+derived,axis=-1)
        missing = ~np.isfinite(x)
        eq(contract['spatial_names'][17:],[n+'_missing' for n in contract['spatial_names'][:17]],'missing names')
        return np.concatenate((np.where(missing,0,x),missing.astype(float)),axis=-1),ids

def first(flags,end=13):
    clipped = flags[...,:end,:]
    frames = np.arange(3,3+end).reshape(1,1,end,1)
    earliest = np.min(np.where(clipped,frames,100),axis=-2)
    return np.where(earliest==100,-1,earliest)

def segments(flags):
    return int(flags[...,0].sum()+((~flags[...,:-1])&flags[...,1:]).sum())

def costs(flags):
    longest = 0
    for row in flags.reshape(-1,13):
        run = 0
        for v in row:
            run = run+1 if v else 0
            longest = max(longest,run)
    return dict(slots=int(flags.sum()),slot_denominator=flags.size,clips=int(flags.any(-1).sum()),
        clip_denominator=int(np.prod(flags.shape[:-1])),segments=segments(flags),longest_run_frames=longest)

def basic(flags,cat):
    contact = cat=='contact';clear = (cat=='clear').all(1)
    passed = (cat=='pass').any(1)&~contact.any(1)
    timely = flags[...,:11,:].any(-2)&contact[:,None,:]
    late = flags[...,11:,:].any(-2)&contact[:,None,:]&~timely
    joint = flags.any(-1);cc = joint[clear];k = flags.shape[1]
    return dict(counts=timely.sum((0,1)).tolist(),late_counts=late.sum((0,1)).tolist(),
        clear_slots=int(cc.sum()),clear_denominator=cc.size,clear_segments=segments(cc),clear_clips=int(cc.any(-1).sum()),
        physical_contact_any_height=int(joint[contact.any(1),:,:11].any(-1).sum()),pass_clips=int(joint[passed].any(-1).sum()),
        contact_event_denominators=(contact.sum(0)*k).tolist(),physical_contact_denominator=int(contact.any(1).sum()*k),
        clear_clip_denominator=int(clear.sum()*k),pass_clip_denominator=int(passed.sum()*k))

def paired(before,after,cat):
    aa,bb = first(before,11),first(after,11);result = []
    for q,height in enumerate(('HEAD','BODY')):
        a,b = aa[cat[:,q]=='contact',:,q],bb[cat[:,q]=='contact',:,q]
        both = (a>=0)&(b>=0);delta = a[both]-b[both]
        result.append(dict(height=height,denominator=a.size,rescue=int(((a<0)&(b>=0)).sum()),
            loss=int(((a>=0)&(b<0)).sum()),both_timely=int(both.sum()),earlier=int((delta>0).sum()),same=int((delta==0).sum()),
            later=int((delta<0).sum()),median_advance_frames=float(np.median(delta)) if len(delta) else None,
            advance_frame_histogram={str(d):int((delta==d).sum()) for d in np.unique(delta)}))
    return result

def describe(grade,ds,refs):
    cat = ds['category'];ans = {name:basic(flags,cat) for name,flags in (('any',grade>0),('strong',grade==2),('light',grade==1))}
    ans.update(paired_any={n:paired(v,grade>0,cat) for n,v in refs.items()},paired_strong={n:paired(v,grade==2,cat) for n,v in refs.items()})
    masks = dict(clear=(cat=='clear').all(1),pass_=(cat=='pass').any(1)&~(cat=='contact').any(1))
    joint = grade.max(-1)
    ans['joint_costs'] = {kind.rstrip('_'):{level:costs((joint==number)[mask]) for level,number in (('light',1),('strong',2))} for kind,mask in masks.items()}
    ts = (grade[...,:11,:]==2).any(-2);tl = (grade[...,:11,:]==1).any(-2)
    ans.update(light_only_timely_contact=((tl&~ts)&(cat=='contact')[:,None,:]).sum((0,1)).tolist(),
        old_alert_query_slots_to_light=int((refs['old_fusion']&(grade==1)).sum()),ordinary_OR_query_slots_to_light=int((refs['ordinary_OR']&(grade==1)).sum()),
        contact_old_timely_to_light_only=((refs['old_fusion'][...,:11,:].any(-2)&tl&~ts)&(cat=='contact')[:,None,:]).sum((0,1)).tolist(),
        groups={name:dict(any=basic((grade>0)[mask],cat[mask]),vs_old_fusion=paired(refs['old_fusion'][mask],(grade>0)[mask],cat[mask])) for name,mask in ds['groups'].items()})
    return ans

def threshold(values,target):
    # Independent tie-group enumeration rather than positional production indexing.
    finite = np.asarray(values).ravel();finite = finite[np.isfinite(finite)]
    if not len(finite) or target>=len(finite): return -np.inf
    uniq,count = np.unique(finite,return_counts=True)
    remain = len(finite)-np.cumsum(count)
    return float(np.nextafter(uniq[np.flatnonzero(remain<=target)[0]],np.inf))

def run():
    began = time.monotonic();audit = OUT/'audit'
    audit.mkdir(exist_ok=True)
    if (audit/'result.json').exists(): raise FileExistsError('Preserve prior audit')
    plan = read(OUT/'PLAN.json');models = read(OUT/'models.json');cuts = read(OUT/'calibrations.json');metrics = read(OUT/'metrics.json')
    contract = read(OUT/'feature_contract.json');ds = load();cal = ds['cal']
    for path,digest in plan['inputs_sha256'].items(): eq(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),digest,'input hash')
    eq(hashlib.sha256((Path(__file__).with_name('cnh_graded_corridor_eval_dev.py')).read_bytes()).hexdigest(),plan['source_sha256'],'producer hash')
    families = {}
    for r in cal['rows']: families.setdefault(r['background_family'],set()).add(r['background_id'])
    assert all(len(v)==2 for v in families.values())
    fit_ids = {min(v) for v in families.values()};cut_ids = {r['background_id'] for r in cal['rows']}-fit_ids
    eq(read(OUT/'cal_partition.json'),dict(fit_background_ids=sorted(fit_ids),calibrate_background_ids=sorted(cut_ids)),'partition')
    assert fit_ids.isdisjoint(cut_ids)
    validation_ids = {r['background_id'] for r in ds['validation']['rows']}
    assert validation_ids.isdisjoint(fit_ids|cut_ids)
    fit = np.array([r['background_id'] in fit_ids for r in cal['rows']]);cut = ~fit
    spatial_data = {}
    for split in ds:
        spatial_data[split],ids = spatial(split,contract)
        np.testing.assert_array_equal(ids,ds[split]['ids'])
    thresholds = read(ROOT/'artifacts.local/work/cnh-graded-evidence-dev-20261010/thresholds.json')
    saved = {};before_first = {};after_first = {};summary = {};gradients = {}
    for split in ds:
        with np.load(OUT/f'{split}_grades.npz',allow_pickle=False) as a: saved[split] = dict(zip(a['keys'].tolist(),a['grades']))
    for seed in SEEDS:
        th = thresholds[str(seed)];tensors = {};refs = {}
        for split,d in ds.items():
            ordinary = smooth(d['raw'][seed]);base = score_features(d['raw'][seed],th['single'])
            tensors[split] = dict(score_only=base,spatial_only=spatial_data[split],score_spatial=np.concatenate((base,spatial_data[split]),axis=-1))
            old = (d['m3']>=.9404184587540165)|(d['local']>=4.625390338985158)
            strong = old|(ordinary>=th['addition']);light = ~strong&(ordinary>=th['single'])
            refs[split] = dict(M3=d['m3']>=.8557642486787612,old_fusion=old,ordinary_OR=strong,ordinary_single=ordinary>=th['single'],prior_light=strong|light)
        for name in plan['models']:
            model = models[f'{seed}/{name}'];beta = np.array(model['beta']);mean = np.array(model['mean']);scale = np.array(model['scale'])
            assert np.isfinite(np.r_[beta,mean,scale]).all() and (scale>0).all()
            eligible = ~refs['cal']['ordinary_OR'][fit]
            count = eligible.sum(-2,keepdims=True)
            weight = np.broadcast_to(np.divide(1.,count,out=np.zeros_like(count,dtype=float),where=count>0),eligible.shape)[eligible]
            weight /= weight.sum();xfit = tensors['cal'][name][fit][eligible]
            y = np.broadcast_to((cal['category'][fit]=='contact')[:,None,None,:],eligible.shape)[eligible]
            expected_mean = np.sum(xfit*weight[:,None],axis=0)
            expected_scale = np.sqrt(np.sum((xfit-expected_mean)**2*weight[:,None],axis=0));expected_scale = np.where(expected_scale>1e-8,expected_scale,1.)
            np.testing.assert_allclose(mean,expected_mean,rtol=0,atol=1e-12);np.testing.assert_allclose(scale,expected_scale,rtol=0,atol=1e-12)
            eq(len(y),model['fit_rows'],'fit rows');eq(int(y.sum()),model['fit_contact_rows'],'fit labels')
            normalized = (xfit-mean)/scale;z = beta[0]+normalized@beta[1:]
            objective = np.sum(weight*(np.logaddexp(0,z)-y*z))+.05*np.dot(beta[1:],beta[1:])
            assert abs(objective-model['objective'])<1e-10
            residual = weight*(1/(1+np.exp(-z))-y);grad = np.r_[residual.sum(),normalized.T@residual+.1*beta[1:]]
            gradients[f'{seed}/{name}'] = float(np.max(np.abs(grad)))
            scores = {split:beta[0]+((tensors[split][name]-mean)/scale)@beta[1:] for split in ds}
            for policy in plan['policies']:
                for fraction in plan['fractions']:
                    key = f'{seed}/{name}/{policy}/{fraction}';record = cuts[key]
                    cat = cal['category'][cut];passed = (cat=='pass').any(1)&~(cat=='contact').any(1);clear = (cat=='clear').all(1)
                    original = refs['cal']['prior_light']&~refs['cal']['ordinary_OR']
                    ep = original if policy=='filter' else ~refs['cal']['ordinary_OR']
                    candidate = np.where(ep[cut],scores['cal'][cut],-np.inf)
                    bp = int(original[cut][passed].any((-1,-2)).sum());bc = int(original[cut][clear].any(-1).sum())
                    tp,tc = int(fraction*bp),int(fraction*bc)
                    theta = max(threshold(candidate[passed].max((-1,-2)),tp),threshold(candidate[clear].max(-1),tc))
                    assert abs(theta-record['theta'])<1e-10,(key,'threshold',theta,record['theta'])
                    flags = ep[cut]&(scores['cal'][cut]>=record['theta'])
                    ap,ac = int(flags[passed].any((-1,-2)).sum()),int(flags[clear].any(-1).sum())
                    eq(record,dict(theta=record['theta'],fraction=fraction,baseline_light_pass_clips=bp,baseline_light_clear_slots=bc,target_light_pass_clips=tp,target_light_clear_slots=tc,actual_light_pass_clips=ap,actual_light_clear_slots=ac,pass_residual=ap-tp,clear_residual=ac-tc),'cal/'+key)
                    assert ap<=tp and ac<=tc
                    for split,d in ds.items():
                        strong = refs[split]['ordinary_OR'];eligible = (refs[split]['prior_light']&~strong) if policy=='filter' else ~strong
                        grade = np.where(strong,2,np.where(eligible&(scores[split]>=record['theta']),1,0)).astype(np.int8)
                        np.testing.assert_array_equal(grade,saved[split][key]);np.testing.assert_array_equal(grade==2,strong)
                        if policy=='filter': assert np.all((grade==0)|refs[split]['prior_light'])
                        report = describe(grade,d,refs[split]);eq(report,metrics[split+'/'+key],'metrics/'+split+'/'+key)
                        summary[(split,key)] = report
                        if split=='validation':
                            before_first[key],after_first[key] = first(refs[split]['prior_light']),first(grade>0)
    rows = 0
    with (OUT/'event_ledger.csv').open(encoding='utf8',newline='') as stream:
        reader = csv.DictReader(stream);lookup = {int(v):i for i,v in enumerate(ds['validation']['ids'])};seen = set()
        for row in reader:
            key = row['key'];i = lookup[int(row['scene'])];k = int(row['replica']);q = ('HEAD','BODY').index(row['height'])
            identity = (key,i,k,q);assert identity not in seen;seen.add(identity)
            eq(int(row['before_first']),int(before_first[key][i,k,q]),'ledger/before');eq(int(row['after_first']),int(after_first[key][i,k,q]),'ledger/after')
            r = ds['validation']['rows'][i]
            eq(row['category'],ds['validation']['category'][i,q],'ledger/category')
            for column,field in (('family','shape_family'),('placement','placement'),('background_family','background_family')): eq(row[column],str(r[field]),'ledger/'+column)
            eq(float(row['rho']),r['rho'],'ledger/rho');rows += 1
    eq(rows,len(saved['validation'])*len(ds['validation']['rows'])*4*2,'complete ledger')
    with (OUT/'summary.csv').open(encoding='utf8',newline='') as stream:
        seen = set()
        for row in csv.DictReader(stream):
            key = '/'.join((row['seed'],row['model'],row['policy'],row['fraction']));identity=(row['split'],key);assert identity not in seen;seen.add(identity)
            r = summary[identity]
            expected = dict(HEAD=r['any']['counts'][0],BODY=r['any']['counts'][1],HEAD_gain_prior=r['paired_any']['prior_light'][0]['rescue'],HEAD_loss_prior=r['paired_any']['prior_light'][0]['loss'],BODY_gain_prior=r['paired_any']['prior_light'][1]['rescue'],BODY_loss_prior=r['paired_any']['prior_light'][1]['loss'],HEAD_extra_strong=r['paired_any']['ordinary_OR'][0]['rescue'],BODY_extra_strong=r['paired_any']['ordinary_OR'][1]['rescue'],clear_slots=r['any']['clear_slots'],clear_clips=r['any']['clear_clips'],pass_clips=r['any']['pass_clips'],light_clear_slots=r['joint_costs']['clear']['light']['slots'],light_pass_slots=r['joint_costs']['pass']['light']['slots'],light_pass_clips=r['joint_costs']['pass']['light']['clips'])
            for field,value in expected.items(): eq(int(row[field]),value,'summary/'+field)
        eq(len(seen),108,'summary cells')
    result = dict(status='PASS',cells=len(summary),model_normalizations_checked=len(models),calibrations=len(cuts),ledger_rows=rows,scalar_checks=CHECKS,
        fit_background_ids=sorted(fit_ids),cutoff_background_ids=sorted(cut_ids),validation_background_ids=sorted(validation_ids),
        model_stationarity_maxabs_gradient=gradients,seconds=time.monotonic()-began,
        independence='Raw cache smoothing, OLS descriptors, scalar prediction, tie-group cutoff, grades, metrics and ledger replayed without production helpers; feature extraction audited separately',
        producer_source_sha256=plan['source_sha256'],audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (audit/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    with threadpool_limits(limits=2): run()
