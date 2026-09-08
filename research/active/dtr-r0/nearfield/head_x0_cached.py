"""HEAD-X0 cached score, error-atlas and support morphology audit; no inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from city_score_separation import curve

REPO=Path(__file__).resolve().parents[4]
WORK=REPO/'artifacts.local/work'
NEAR=REPO/'artifacts.local/nearfield'
HASHES={}


def sha(path):
    path=Path(path)
    with path.open('rb') as stream: value=hashlib.file_digest(stream,'sha256').hexdigest()
    HASHES[str(path.resolve())]=value
    return value


def read(path):
    sha(path)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def cached_array(root,record):
    path=(root/record['path']).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or sha(path)!=record['sha256']:
        raise ValueError('Array path/hash mismatch')
    return np.load(path,mmap_mode='r',allow_pickle=False)


def load_partition(root,split,spec_path):
    manifest=read(root/'manifest.json')
    if manifest['status']!='PASS':raise ValueError('Cache not accepted')
    entry=manifest['partitions'][split]
    labels=read(root/('supervision' if split=='train' else 'evaluator')/f'{split}.json')
    if entry['sample_indices']!=labels['sample_indices']:raise ValueError('Label order mismatch')
    spec=read(spec_path)
    result=dict(ids=entry['sample_indices'],rgb=cached_array(root,entry['rgb']),
        near=cached_array(root,labels['near']),support=cached_array(root,labels['support']))
    result['cases']=[spec['cases'][i] for i in result['ids']]
    if result['rgb'].shape!=(len(result['ids']),144,256,3) or result['support'].shape!=(len(result['ids']),2,18,32):
        raise ValueError('Cache shape mismatch')
    return result


def predictions(path,partition):
    sha(path)
    with np.load(path,allow_pickle=False) as archive:
        ids=archive['sample_indices'].tolist()
        if ids!=partition['ids']:raise ValueError('Prediction/cache order mismatch: '+str(path))
        near=archive['near'].copy(); support=archive['support'].copy()
    if near.shape!=(len(ids),2) or support.shape!=(len(ids),2,18,32):raise ValueError('Prediction shape')
    if not np.isfinite(near).all() or not np.isfinite(support).all():raise ValueError('Nonfinite probabilities')
    if not ((near>=0)&(near<=1)).all() or not ((support>=0)&(support<=1)).all():raise ValueError('Invalid probabilities')
    return near,support


def quantiles(values):
    if not len(values):return None
    return {str(q):float(v) for q,v in zip((0,.05,.25,.5,.75,.95,1),np.quantile(values,(0,.05,.25,.5,.75,.95,1)))}


def score_summary(scores,labels):
    scores=np.asarray(scores,dtype=np.float64); labels=np.asarray(labels)
    known=labels>=0
    result=curve(scores[known].tolist(),labels[known].astype(int).tolist())
    result['unknown_count']=int((~known).sum())
    result['classes']={}
    for name,value in (('positive',1),('negative',0)):
        selected=scores[labels==value]
        clipped=np.clip(selected,np.finfo(np.float64).eps,1-np.finfo(np.float64).eps)
        reconstructed=np.log(clipped)-np.log1p(-clipped)
        result['classes'][name]=dict(n=len(selected),probability_quantiles=quantiles(selected),
            reconstructed_logit_quantiles=quantiles(reconstructed),
            saturated_zero_count=int((selected==0).sum()),saturated_one_count=int((selected==1).sum()))
    result['logit_scope']='Inverse sigmoid reconstructed from cached probabilities, NOT raw logits; float64 epsilon clips saturation.'
    return result


def distribution_supplement(out):
    """Plot six F distributions from cached scores; leave prior audit untouched."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    started=time.perf_counter()
    destination=out/'head-score-six-distributions.png'
    supplement=out/'distribution-supplement.json'
    if destination.exists() or supplement.exists():raise FileExistsError('Distribution supplement exists')
    prior=read(out/'receipt.json')
    if prior['status']!='PASS':raise ValueError('Completed original audit required')
    oldroot=NEAR/'city-gate-intervention-20260908/inputs/collection500-training-cache-v1'
    oldspec=NEAR/'city-pcg-20260908/worker-scale500-v1/collection500-evidence/collection500-v1/evaluator/spec.json'
    parts={
        'old750':load_partition(oldroot,'train',oldspec),
        'new384':load_partition(WORK/'city-relational-train-20260908/evidence/cache','train',WORK/'city-relational-train-20260908/evidence/capture/evaluator/spec.json'),
        'DEV128':load_partition(WORK/'city-dev-baseline-20260908/evidence/cache','dev',WORK/'city-dev-baseline-20260908/evidence/capture/evaluator/spec.json'),
        'plaza750':load_partition(oldroot,'test',oldspec)}
    run=WORK/'city-full-fit-20260908/model-run-v1'
    predictions_by_part={name:predictions(run/f'{filename}-predictions.npz',parts[name])[0][:,1].astype(np.float64)
                         for name,filename in (('old750','old750'),('new384','new384'),('DEV128','dev'),('plaza750','plaza'))}
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    counts={}
    probability_bins=np.linspace(0,1,26)
    all_scores=np.concatenate(list(predictions_by_part.values()))
    eps=np.finfo(np.float64).eps
    clipped=np.clip(all_scores,eps,1-eps)
    all_reconstructed=np.log(clipped)-np.log1p(-clipped)
    logit_bins=np.linspace(np.floor(all_reconstructed.min()),np.ceil(all_reconstructed.max()),33)
    for column,(domain,members) in enumerate((('TRAIN1134',['old750','new384']),('DEV128',['DEV128']),('plaza750',['plaza750']))):
        scores=np.concatenate([predictions_by_part[name] for name in members])
        labels=np.concatenate([parts[name]['near'][:,1] for name in members])
        counts[domain]={}
        for classname,value,color in (('positive',1,'#d64b40'),('negative',0,'#2976b8')):
            selected=scores[labels==value]
            bounded=np.clip(selected,eps,1-eps)
            reconstructed=np.log(bounded)-np.log1p(-bounded)
            counts[domain][classname]=dict(n=len(selected),probability_median=float(np.median(selected)),
                saturation_zero=int((selected==0).sum()),saturation_one=int((selected==1).sum()))
            for row,values,bins in ((0,selected,probability_bins),(1,reconstructed,logit_bins)):
                axes[row,column].hist(values,bins=bins,density=True,histtype='step',linewidth=2,
                    color=color,label=f'{classname} n={len(selected)}')
                axes[row,column].axvline(np.median(values),color=color,linestyle=':',linewidth=1)
                axes[row,column].set_ylabel('Within-class density')
                axes[row,column].grid(alpha=.2)
            axes[0,column].set_title(domain)
        axes[0,column].set(xlim=(0,1),xlabel='Cached HEAD probability')
        axes[1,column].set(xlim=(logit_bins[0],logit_bins[-1]),xlabel='Reconstructed logit (not raw)')
        axes[0,column].legend(fontsize=9)
    fig.suptitle('F2000 HEAD: six class-conditional cached distributions\nDotted lines: class medians; consumed descriptive audit, no threshold selection',fontsize=13)
    fig.text(.5,.012,'Reconstructed logit = inverse sigmoid of cached probabilities; saturated values clipped at float64 epsilon. Each class integrates to one; prevalence differs.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.93));fig.savefig(destination,dpi=170);plt.close(fig)
    source=Path(__file__)
    sha(source)
    write(supplement,dict(status='PASS',backend='CPU',backend_reason='TASK_NOT_GPU_SUITABLE_SMALL_CACHED_ARRAYS_AND_PLOT',
        optimizer_steps=0,inference_calls=0,original_receipt_sha256=sha(out/'receipt.json'),
        source_sha256=sha(source),plot_path=str(destination),plot_sha256=sha(destination),
        input_sha256=dict(HASHES),counts=counts,probability_bins=probability_bins.tolist(),reconstructed_logit_bins=logit_bins.tolist(),
        density='Each class normalized separately within each domain; six distributions, shown in two transforms',
        seconds=time.perf_counter()-started))
    print(json.dumps(dict(status='PASS',plot=str(destination),supplement=str(supplement))),flush=True)


def morphology(mask,rgb,known):
    ys,xs=np.where(mask); n=len(xs)
    if not n:return dict(area_cells=0,bbox=None,aspect=None,centroid=None,pca_orientation_deg=None,thickness_proxy_cells=None)
    x0,x1,y0,y1=int(xs.min()),int(xs.max()),int(ys.min()),int(ys.max())
    covariance=np.cov(np.stack([xs,ys])) if n>1 else np.zeros((2,2))
    vals,vecs=np.linalg.eigh(covariance)
    orientation=float(np.degrees(np.arctan2(vecs[1,-1],vecs[0,-1]))) if n>1 and vals[-1]>0 else None
    if orientation is not None:orientation=(orientation+90)%180-90
    # Padding makes distance to background defined even for full-edge masks.
    distance=ndimage.distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]
    gray=np.asarray(Image.fromarray(rgb).convert('L').resize((32,18),Image.Resampling.BOX),dtype=float)
    ring=ndimage.binary_dilation(mask,iterations=1)&~mask&known
    contrast=float(gray[mask].mean()-gray[ring].mean()) if ring.any() else None
    return dict(area_cells=n,bbox=[x0,y0,x1+1,y1+1],bbox_width=x1-x0+1,bbox_height=y1-y0+1,
        aspect=(x1-x0+1)/(y1-y0+1),centroid=[float(xs.mean()),float(ys.mean())],
        centroid_normalized=[float((xs.mean()+.5)/32),float((ys.mean()+.5)/18)],top_distance_cells=y0,
        pca_orientation_deg=orientation,pca_major_variance=float(vals[-1]),pca_minor_variance=float(vals[0]),
        thickness_proxy_cells=float(2*distance[mask].mean()),
        rgb_gray_support_mean=float(gray[mask].mean()),rgb_gray_known_ring_mean=float(gray[ring].mean()) if ring.any() else None,
        rgb_contrast_signed=contrast,rgb_contrast_absolute=abs(contrast) if contrast is not None else None,
        scope='Pooled visible query support in32x18, not full-object geometry or physical thickness; PCA image y increases downward.')


def patch_at(rgb,peak):
    y,x=peak; px=int((x+.5)*8); py=int((y+.5)*8)
    padded=np.pad(rgb,((16,16),(16,16),(0,0)),mode='edge')
    patch=padded[py:py+32,px:px+32]
    gray=np.asarray(Image.fromarray(patch).convert('L'),dtype=np.float64)/255
    edge=np.hypot(ndimage.sobel(gray,axis=0),ndimage.sobel(gray,axis=1))
    small=np.asarray(Image.fromarray(patch).resize((8,8),Image.Resampling.BOX),dtype=np.float64).ravel()/255
    edge_small=np.asarray(Image.fromarray(edge.astype(np.float32),mode='F').resize((8,8),Image.Resampling.BOX)).ravel()
    feature=np.concatenate([small,edge_small,[gray.mean(),gray.std(),edge.mean(),edge.std()]])
    return patch,feature,dict(model_rgb_center_xy=[px,py],patch_size=32,edge_padding=True,
        gray_mean=float(gray.mean()),gray_std=float(gray.std()),sobel_mean=float(edge.mean()),sobel_std=float(edge.std()))


def image_tile(rgb,prob,gt,title,description):
    peak=np.unravel_index(int(prob.argmax()),prob.shape)
    patch,_,_=patch_at(rgb,peak)
    heat=np.stack([prob,np.zeros_like(prob),1-prob],axis=-1)
    heat=np.asarray(Image.fromarray((heat*255).astype(np.uint8)).resize((256,144),Image.Resampling.NEAREST))
    overlay=(.55*rgb+.45*heat).astype(np.uint8)
    truth=rgb.copy(); enlarged=np.asarray(Image.fromarray(gt.astype(np.int32)).resize((256,144),Image.Resampling.NEAREST))
    truth[enlarged==1]=(truth[enlarged==1]*.35+np.array([0,255,0])*.65).astype(np.uint8)
    truth[enlarged==-1]=(truth[enlarged==-1]*.65+np.array([110,70,150])*.35).astype(np.uint8)
    canvas=Image.new('RGB',(1024,198),'#171b23'); draw=ImageDraw.Draw(canvas)
    for k,panel in enumerate((rgb,overlay,truth)):
        canvas.paste(Image.fromarray(panel),(k*256,22))
    draw.ellipse((256+peak[1]*8+1,22+peak[0]*8+1,256+peak[1]*8+7,22+peak[0]*8+7),outline='white',width=2)
    canvas.paste(Image.fromarray(patch).resize((128,128),Image.Resampling.NEAREST),(784,30))
    draw.text((4,2),title,fill='white')
    draw.text((4,174),'RGB model input | HEAD probability overlay + peak | GT green / UNKNOWN purple | 32px peak patch',fill='#cccccc')
    draw.text((4,186),description,fill='#eeeeee')
    return canvas


def contacts(paths,out,prefix,per_page=8):
    outputs=[]
    for start in range(0,len(paths),per_page):
        subset=paths[start:start+per_page]
        sheet=Image.new('RGB',(2048,198*((len(subset)+1)//2)),'#171b23')
        for i,path in enumerate(subset):
            with Image.open(path) as im:sheet.paste(im,((i%2)*1024,(i//2)*198))
        dest=out/f'{prefix}-{start//per_page+1:02d}.png'; sheet.save(dest); outputs.append(str(dest))
    return outputs


def main(out):
    started=time.perf_counter()
    if (out/'receipt.json').exists():raise FileExistsError(out/'receipt.json')
    out.mkdir(parents=True,exist_ok=True)
    oldroot=NEAR/'city-gate-intervention-20260908/inputs/collection500-training-cache-v1'
    oldspec=NEAR/'city-pcg-20260908/worker-scale500-v1/collection500-evidence/collection500-v1/evaluator/spec.json'
    partitions={
        'old750':load_partition(oldroot,'train',oldspec),
        'new384':load_partition(WORK/'city-relational-train-20260908/evidence/cache','train',WORK/'city-relational-train-20260908/evidence/capture/evaluator/spec.json'),
        'DEV128':load_partition(WORK/'city-dev-baseline-20260908/evidence/cache','dev',WORK/'city-dev-baseline-20260908/evidence/capture/evaluator/spec.json'),
        'plaza750':load_partition(oldroot,'test',oldspec)}
    pred={}; ranking={}
    for arm,run in (('F2000',WORK/'city-full-fit-20260908/model-run-v1'),('B2000',WORK/'city-fit-adequacy-20260908/model-run-v1')):
        pred[arm]={}
        for name,filename in (('old750','old750'),('new384','new384'),('DEV128','dev'),('plaza750','plaza')):
            pred[arm][name]=predictions(run/f'{filename}-predictions.npz',partitions[name])
        for name,members in (('TRAIN1134',['old750','new384']),('DEV128',['DEV128']),('plaza750',['plaza750'])):
            ranking[f'{arm}/{name}']=score_summary(np.concatenate([pred[arm][m][0][:,1] for m in members]),np.concatenate([partitions[m]['near'][:,1] for m in members]))
        # Also expose the old/new TRAIN mixture, rather than hiding its source difference.
        for name in ('old750','new384'):
            ranking[f'{arm}/{name}']=score_summary(pred[arm][name][0][:,1],partitions[name]['near'][:,1])
    original=NEAR/'city-pcg-20260908/worker-finetune-prep-v1/run-v1/baseline-city-predictions.npz'
    # Legacy original cache lacks sample_indices: do not infer identity by row count.
    original_gap='Legacy original cache has no sample_indices; original comparison omitted pending explicit identity receipt.'
    write(out/'ranking.json',dict(status='PASS',results=ranking,original_gap=original_gap,scope='Consumed cached HEAD ranking; no inference or threshold selection.'))
    print(json.dumps(dict(phase='RANKING_READY',path=str(out/'ranking.json'),summary={k:{m:v[m] for m in ('roc_auc','average_precision','positives','negatives')} for k,v in ranking.items()})),flush=True)
    selection=read(WORK/'city-full-fit-20260908/model-run-v1/selection.json')
    threshold=selection['heads']['HEAD']['threshold']
    p=partitions['plaza750']; n,s=pred['F2000']['plaza750']
    errors=[]; features=[]; tiles=[]; tile_dir=out/'error-tiles';tile_dir.mkdir(exist_ok=True)
    for i in range(len(p['ids'])):
        label=int(p['near'][i,1]); decision=float(n[i,1])>=threshold
        if label<0 or decision==bool(label):continue
        kind='FP' if decision else 'FN'; rgb=np.array(p['rgb'][i]); gt=np.array(p['support'][i,1]); prob=s[i,1].astype(np.float64)
        known=gt>=0; active=(prob>=.5)&known
        components,count=ndimage.label(active,structure=np.ones((3,3)))
        sizes=np.bincount(components.ravel())[1:]
        peak=np.unravel_index(int(prob.argmax()),prob.shape)
        patch,feature,patch_stats=patch_at(rgb,peak)
        case=p['cases'][i]
        row=dict(position=i,sample_index=p['ids'][i],kind=kind,group_id=case.get('group_id'),
            family=case.get('condition',{}).get('family'),mesh_assets=[o.get('mesh_asset') for o in case.get('objects',[]) if o.get('mesh_asset')],
            score=float(n[i,1]),threshold=threshold,support_max=float(prob.max()),support_mean=float(prob.mean()),
            support_known_mean=float(prob[known].mean()) if known.any() else None,
            connected_components_8=int(count),largest_component_cells=int(sizes.max()) if len(sizes) else 0,
            active_known_cells=int(active.sum()),active_unknown_cells=int(((prob>=.5)&~known).sum()),
            peak_xy=[int(peak[1]),int(peak[0])],peak_gt_value=int(gt[peak]),
            predicted_mask=morphology(active,rgb,known),gt_mask=morphology(gt==1,rgb,known),peak_patch=patch_stats)
        path=tile_dir/f'{kind}-{p["ids"][i]:04d}.png'
        tile=image_tile(rgb,prob,gt,f'{kind} index={p["ids"][i]} score={row["score"]:.5f} threshold={threshold:.5f}',
            f'GTcells={row["gt_mask"]["area_cells"]} active={row["active_known_cells"]} CC8={count} peakGT={row["peak_gt_value"]} family={row["family"] or "UNAVAILABLE"}')
        tile.save(path);row['tile_path']=str(path);errors.append(row);features.append(feature);tiles.append(path)
    assert Counter(r['kind'] for r in errors)=={'FP':171,'FN':15},'Frozen error count mismatch'
    error_pages=contacts(tiles,out,'all-errors')
    scaled=StandardScaler().fit_transform(np.stack(features))
    clustering=KMeans(n_clusters=4,random_state=17,n_init=10).fit(scaled)
    clusters=[];medoid_tiles=[]
    for cluster in range(4):
        positions=np.flatnonzero(clustering.labels_==cluster)
        distance=pairwise_distances(scaled[positions],metric='euclidean')
        central=positions[np.argsort(distance.sum(axis=1),kind='stable')[:3]]
        for i in positions:errors[int(i)]['patch_cluster']=cluster
        medoids=[dict(error_position=int(i),sample_index=errors[int(i)]['sample_index'],kind=errors[int(i)]['kind'],tile_path=errors[int(i)]['tile_path']) for i in central]
        cluster_tiles=[]
        for i in central:
            path=Path(errors[int(i)]['tile_path'])
            with Image.open(path) as im:
                canvas=im.copy();ImageDraw.Draw(canvas).text((785,162),f'CLUSTER {cluster}',fill='white')
                decorated=out/f'cluster-{cluster}-medoid-{int(i):03d}.png';canvas.save(decorated);cluster_tiles.append(decorated)
        medoid_tiles.extend(cluster_tiles)
        clusters.append(dict(cluster=cluster,count=len(positions),error_counts=dict(Counter(errors[int(i)]['kind'] for i in positions)),medoids=medoids))
    medoid_pages=contacts(medoid_tiles,out,'cluster-medoids',per_page=12)
    write(out/'errors.json',dict(status='PASS',threshold=threshold,rows=errors,atlas_pages=error_pages))
    write(out/'clusters.json',dict(status='PASS',k=4,seed=17,n_init=10,
        features='8x8 RGB,8x8 Sobel magnitude, patch gray/edge mean/std; StandardScaler; fixed32px peak patch with edge padding',
        representative_rule='Three actual members with smallest summed pairwise Euclidean distances within cluster; ties source order',
        scope='Descriptive low-level patch clusters, NOT semantic classes; no learned model inference',clusters=clusters,contact_pages=medoid_pages))
    morphology_rows=[];positive_tiles=[];positive_dir=out/'plaza-positive-tiles';positive_dir.mkdir(exist_ok=True)
    for name,part in partitions.items():
        for i in np.flatnonzero(part['near'][:,1]==1):
            rgb=np.array(part['rgb'][i]);gt=np.array(part['support'][i,1]);case=part['cases'][i]
            row=dict(partition=name,sample_index=part['ids'][i],group_id=case.get('group_id'),
                family=case.get('condition',{}).get('family'),
                mesh_assets=[o.get('mesh_asset') for o in case.get('objects',[]) if o.get('mesh_asset')],
                morphology=morphology(gt==1,rgb,gt>=0))
            morphology_rows.append(row)
            if name=='plaza750':
                path=positive_dir/f'positive-{part["ids"][i]:04d}.png'
                image_tile(rgb,pred['F2000'][name][1][i,1],gt,f'ALL plaza HEAD positive index={part["ids"][i]}',
                    f'GTcells={row["morphology"]["area_cells"]} aspect={row["morphology"]["aspect"]} mesh={";".join(row["mesh_assets"])}').save(path)
                positive_tiles.append(path)
    assert len(positive_tiles)==15
    positive_pages=contacts(positive_tiles,out,'all-plaza-head-positive')
    summaries={}
    for name,members in (('TRAIN1134',['old750','new384']),('DEV128',['DEV128']),('plaza750',['plaza750']),('old750',['old750']),('new384',['new384'])):
        selected=[r for r in morphology_rows if r['partition'] in members]
        summaries[name]=dict(n=len(selected),families=dict(Counter(r['family'] or 'UNAVAILABLE' for r in selected)),
            quantiles={field:quantiles(np.array([r['morphology'][field] for r in selected if r['morphology'].get(field) is not None]))
                for field in ('area_cells','aspect','top_distance_cells','pca_orientation_deg','thickness_proxy_cells','rgb_contrast_absolute')})
    write(out/'morphology.json',dict(status='PASS',rows=morphology_rows,summary=summaries,plaza_positive_pages=positive_pages,
        scope='HEAD positive native visible query support pooled32x18; not object identity, full silhouette, physical thickness, or model-independent natural evidence.'))
    sha(Path(__file__))
    write(out/'receipt.json',dict(status='PASS',backend='CPU',backend_reason='TASK_NOT_GPU_SUITABLE_SMALL_CACHED_ARRAYS_AND_REPORTS',
        optimizer_steps=0,inference_calls=0,errors=dict(Counter(r['kind'] for r in errors)),
        input_sha256=HASHES,seconds=time.perf_counter()-started,outputs=dict(ranking='ranking.json',errors='errors.json',clusters='clusters.json',morphology='morphology.json'),
        rgb_scope='Exact cached144x256 model inputs; tiles enlarge/overlay solely for review',
        atlas_pages=error_pages,cluster_pages=medoid_pages,positive_pages=positive_pages))
    print(json.dumps(dict(phase='COMPLETE',output=str(out),seconds=time.perf_counter()-started,cluster_pages=medoid_pages,positive_pages=positive_pages)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=WORK/'head-x0-20260908/cached-v1')
    parser.add_argument('--distribution-only',action='store_true')
    args=parser.parse_args()
    (distribution_supplement if args.distribution_only else main)(args.output.resolve())
