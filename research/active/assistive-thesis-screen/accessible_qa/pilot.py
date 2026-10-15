"""CPU-only exploratory diagnostics. Features are question text or raw pixels only.

Annotations are labels/evaluation only. No test-split annotations are opened.
Image sample: 96 deterministic SHA256-ranked filenames, 64 development-train and
32 development-eval. This is not a source/photographer-disjoint confirmation.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import collections, concurrent.futures, csv, hashlib, json, math, re, time, urllib.request
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[4]/'artifacts.local/work/assistive-thesis-screen-20261015/accessible_qa'
cv2.setNumThreads(1)

def load(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))
def norm(q): return ' '.join(re.findall(r"[a-z0-9]+",q.lower()))
def tokens(q): return re.findall(r"[a-z0-9]+",q.lower())
def metrics(y,score):
    y=np.array(y,dtype=int); score=np.asarray(score); n=len(y); pos=int(y.sum()); neg=n-pos
    # AP uses score ties as one threshold, matching the standard noninterpolated AP.
    order=np.argsort(-score,kind='stable'); ys=y[order]; ss=score[order]
    tp=fp=0; ap=0.; auc=0.
    for start in range(n):
        if start and ss[start]==ss[start-1]: continue
        end=start+1
        while end<n and ss[end]==ss[start]: end+=1
        dp=int(ys[start:end].sum()); dn=end-start-dp
        tp+=dp; fp+=dn
        if pos: ap+=dp/pos*tp/(tp+fp)
        # Number of positive-negative pairs in score descending order.
        auc+=dp*(neg-fp+0.5*dn)
    pred=score>=.5
    return {'n':n,'positive':pos,'negative':neg,'prevalence':pos/n if n else None,
            'auroc':auc/(pos*neg) if pos*neg else None,'average_precision':ap if pos else None,
            'accuracy_at_0_5':float((pred==y).mean()),'true_positive':int((pred&(y==1)).sum()),
            'false_positive':int((pred&(y==0)).sum())}

def question_pilot(train,val,quality):
    # Explicitly construct the permitted input records before touching labels.
    train_q=[r['question'] for r in train]; val_q=[r['question'] for r in val]
    y=np.array([1-int(r['answerable']) for r in train]); vy=[1-int(r['answerable']) for r in val]
    counts=[collections.Counter(),collections.Counter()]; documents=collections.Counter()
    for q in train_q: documents.update(set(tokens(q)))
    vocab={w for w,n in documents.items() if n>=3}
    for q,label in zip(train_q,y): counts[label].update(w for w in tokens(q) if w in vocab)
    totals=[sum(c.values())+len(vocab) for c in counts]
    log_prior=math.log((y.sum()+1)/(len(y)-y.sum()+1))
    weights={w:math.log((counts[1][w]+1)/totals[1])-math.log((counts[0][w]+1)/totals[0]) for w in vocab}
    def predict(q):
        z=log_prior+sum(weights.get(w,0.) for w in tokens(q))
        return 1/(1+math.exp(-max(-50,min(50,z))))
    scores=[predict(q) for q in val_q]
    known=set(map(norm,train_q)); novel=[i for i,q in enumerate(val_q) if norm(q) not in known]
    qmap={r['image']:r for r in quality}
    strata=collections.defaultdict(list)
    for r in val:
        q=qmap.get(r['image'])
        if q:
            strata['unrecognizable>=3' if q['unrecognizable']>=3 else 'recognizable<3'].append(r)
            flaws=q['flaws']
            active=any(flaws[k]>=3 for k in ['BLR','BRT','DRK','FRM','OBS','ROT'])
            strata['quality_flaw>=3' if active else 'no_majority_quality_flaw'].append(r)
    summary={
        'train_n':len(train),'eval_n':len(val),'features':'question unigram counts ONLY; no answers, answer_type, quality labels, image IDs',
        'baseline':'multinomial naive Bayes, min_df=3, alpha=1; positive=unanswerable',
        'vocabulary_size':len(vocab),'prior_only':metrics(vy,[float(y.mean())]*len(val)),
        'question_only':metrics(vy,scores),'normalized_question_unseen_subset':metrics([vy[i] for i in novel],[scores[i] for i in novel]),
        'normalized_question_seen_n':len(val)-len(novel),
        'strata':{k:{'n':len(v),'unanswerable':sum(not r['answerable'] for r in v),'rate':sum(not r['answerable'] for r in v)/len(v)} for k,v in strata.items()},
        'unanswerable_vote_histogram':dict(sorted(collections.Counter(sum(a['answer'].strip().lower() in {'unanswerable','unsuitable','unsuitable image'} for a in r['answers']) for r in val).items())),
        'mixed_answerable_unanswerable_votes':sum(0<sum(a['answer'].strip().lower() in {'unanswerable','unsuitable','unsuitable image'} for a in r['answers'])<len(r['answers']) for r in val),
        'read_text_question_proxy': {'regex':'read|say|written|label|expir|text|writing',
          'n':sum(bool(re.search(r'read|say|written|label|expir|text|writing',r['question'],re.I)) for r in val)},
    }
    with (ROOT/'question_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['image','unanswerable_label','unanswerable_probability','normalized_question_seen'])
        w.writerows((r['image'],vy[i],scores[i],norm(r['question']) in known) for i,r in enumerate(val))
    return summary

def fetch_image(r):
    path=ROOT/'images'/r['file_name']; path.parent.mkdir(exist_ok=True)
    try:
        if not path.exists():
            with urllib.request.urlopen(r['vizwiz_url'],timeout=15) as response:
                data=response.read(4_000_001)
            if len(data)>4_000_000: raise RuntimeError('image byte ceiling exceeded')
            path.write_bytes(data)
        gray=cv2.imdecode(np.frombuffer(path.read_bytes(),dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        if gray is None: raise RuntimeError('decode failed')
        height,width=gray.shape
        gray=cv2.resize(gray,(256,256),interpolation=cv2.INTER_AREA).astype(np.float32)
        lap=cv2.Laplacian(gray,cv2.CV_32F); dx=cv2.Sobel(gray,cv2.CV_32F,1,0);dy=cv2.Sobel(gray,cv2.CV_32F,0,1)
        hist=np.histogram(gray,bins=32,range=(0,256))[0]/gray.size
        feats=[gray.mean(),gray.std(),np.mean(gray<20),np.mean(gray>235),math.log1p(lap.var()),
               np.mean(np.hypot(dx,dy)),float(-(hist*np.log(hist+1e-12)).sum()),width/height]
        # Raw low-frequency spatial brightness variation, no semantic or label features.
        for y in range(2):
            for x in range(2):
                tile=gray[y*128:(y+1)*128,x*128:(x+1)*128]
                feats.extend([tile.mean(),tile.std()])
        return {'image':r['file_name'],'url':r['vizwiz_url'],'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'features':[float(v) for v in feats],'error':None}
    except Exception as exc: return {'image':r['file_name'],'url':r['vizwiz_url'],'error':str(exc)}

def image_pilot(products):
    selected=sorted(products,key=lambda r:hashlib.sha256(('aqa-v1:'+r['file_name']).encode()).hexdigest())[:96]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: receipts=list(pool.map(fetch_image,selected))
    (ROOT/'image_receipts.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
    valid=[(r,v) for r,v in zip(selected,receipts) if not v['error']]
    if len(valid)!=96: return {'selected_n':96,'downloaded_n':len(valid),'failures':[r for r in receipts if r['error']],'status':'INCOMPLETE_IMAGE_SAMPLE_NO_BASELINE'}
    x=np.array([v['features'] for r,v in valid]); mu=x[:64].mean(0);sd=x[:64].std(0)+1e-6;x=(x-mu)/sd
    x=np.column_stack([np.ones(len(x)),x]); reg=np.eye(x.shape[1])*5.;reg[0,0]=1e-6
    results={}
    predictions=[]
    for target in ['low_quality','blur','framing']:
        y=np.array([int(r['image_type']=='low-quality') if target=='low_quality' else int(r[target]) for r,v in valid])
        # Ridge score is a transparent weak classifier; closed-form fit avoids tuning.
        weights=np.linalg.solve(x[:64].T@x[:64]+reg,x[:64].T@y[:64]); score=x[64:]@weights
        results[target]={'training_positive':int(y[:64].sum()),'prior_only':metrics(y[64:],[float(y[:64].mean())]*32),'pixel_ridge':metrics(y[64:],score)}
        if target=='blur': results[target]['laplacian_variance_baseline']=metrics(y[64:],-np.array([v['features'][4] for r,v in valid])[64:])
        for i,s in enumerate(score): predictions.append({'image':valid[i+64][0]['file_name'],'target':target,'label':int(y[i+64]),'score':float(s)})
    (ROOT/'image_predictions.json').write_text(json.dumps(predictions,indent=2),encoding='utf-8')
    return {'selected_n':96,'train_n':64,'eval_n':32,'sample':'SHA256(aqa-v1:filename) order; public VizWiz train product images only',
            'features':'16 pixel-derived grayscale quality features; no OCR/semantic model; no label or answer features',
            'model':'ridge regression lambda=5 fixed before eval; threshold 0.5; no evaluation tuning',
            'downloaded_bytes':sum(r['bytes'] for r in receipts),'targets':results}

def main():
    start=time.monotonic(); cpu=time.process_time()
    train=load('vqa_annotations/train.json');val=load('vqa_annotations/val.json');quality=load('quality_annotations/val.json');products=load('products.json')
    result={'evidence':'EXPLORE reused public train/validation; no protected test access','question':question_pilot(train,val,quality)}
    result['product_annotation_description']={'n':len(products),'unrecognizable':sum(r['unrecognizable'] for r in products),
       'low_quality':sum(r['image_type']=='low-quality' for r in products),
       'nonempty_annotation_fields':{k:sum(bool(r[k]) for r in products) for k in ['product','brand','variety']},
       'interpretation':'Nonempty field is annotation availability, NOT established answerability; this selected product set has no unrecognizable images.'}
    result['images']=image_pilot(products)
    result['wall_seconds']=time.monotonic()-start;result['cpu_seconds']=time.process_time()-cpu
    (ROOT/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
