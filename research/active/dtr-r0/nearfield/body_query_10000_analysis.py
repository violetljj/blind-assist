"""Independent confusion/AUC check for the accepted 10000-frame B comparison."""
import argparse
from pathlib import Path
import numpy as np
from body_query_data import read, write, truth, sha


def auc(score, label):
    negatives = np.sort(score[label == 0]); positives = score[label == 1]
    return float((np.searchsorted(negatives,positives,'left') +
                 np.searchsorted(negatives,positives,'right')).sum() / (2*len(negatives)*len(positives)))


def confusion(score, label, threshold):
    p = score.astype(np.float64) >= threshold; pos = label == 1
    return dict(TP=int((p & pos).sum()), FP=int((p & ~pos).sum()),
                FN=int((~p & pos).sum()), TN=int((~p & ~pos).sum()))


def low_fp(score,label,fraction):
    # Diagnostic curve only. Sorting whole equal-score blocks preserves ties.
    order = np.argsort(-score,kind='stable'); s=score[order]; y=label[order]
    ends=np.r_[np.flatnonzero(s[:-1] != s[1:]),len(s)-1]
    tp=np.cumsum(y==1)[ends];fp=np.cumsum(y==0)[ends]
    valid=fp <= int((label==0).sum()*fraction)
    return dict(TP=int(tp[valid].max(initial=0)), positives=int((label==1).sum()),
                FP_budget=int((label==0).sum()*fraction), FPR_budget=fraction)


def run(cache,out):
    result=read(out/'result.json');selection=read(out/'selection.json');receipt=read(out/'receipt.json')
    protocol=read(out/'protocol.json');fit=read(out/'fit-complete.json')
    assert receipt['fits']==1 and receipt['steps']==fit['steps']==2000
    assert sha(cache/'manifest.json') == protocol['cache_sha256']
    assert sha(Path(__file__).with_name('body_query_10000_train.py')) == protocol['source_sha256']
    assert sha(Path(__file__).with_name('BODY_QUERY_10000_B_PROTOCOL_20260909.md')) == protocol['brief_sha256']
    for name,digest in protocol['dependency_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    assert sha(out/'NEW-step2000.pt') == fit['checkpoint_sha256'] == selection['NEW']['checkpoint_sha256']
    assert sha(out/'result.json') == receipt['result_sha256']
    assert sha(out/'selection.json') == receipt['selection_sha256']
    summary={}; verified=0
    for role in ('train','dev','eval'):
        rec,gt=truth(cache,role);summary[role]={};decisions={}
        for arm in ('OLD','NEW'):
            pred=dict(np.load(out/f'{arm}-{role}.npz',allow_pickle=False));heads={}
            cutoffs=selection[arm]['thresholds'];decisions[arm]=pred['near'].astype(np.float64)>=cutoffs
            for h,name in enumerate(('BODY','HEAD')):
                c=confusion(pred['near'][:,h],gt['near'][:,h],cutoffs[h]);a=auc(pred['near'][:,h],gt['near'][:,h])
                saved=result['arms'][arm][role]['heads'][name]['selected']
                assert all(c[k] == saved[k] for k in c) and abs(a-saved['AUC'])<1e-12
                heads[name]=dict(**c,AUC=a,low_fp={str(f):low_fp(pred['near'][:,h],gt['near'][:,h],f) for f in (.05,.10)})
                verified+=1
            sliced={}
            for key in ('region_id','family','condition'):
                sliced[key]={}
                for value in dict.fromkeys(r[key] for r in rec['records']):
                    ids=np.array([i for i,r in enumerate(rec['records']) if r[key]==value])
                    sliced[key][value]={name:confusion(pred['near'][ids,h],gt['near'][ids,h],cutoffs[h]) for h,name in enumerate(('BODY','HEAD'))}
            summary[role][arm]=dict(heads=heads,slices=sliced,groups=result['arms'][arm][role]['selected_groups']['correct'],
                total_groups=result['arms'][arm][role]['selected_groups']['total'], query_strata=result['arms'][arm][role]['query_strata'])
        paired={}
        for h,name in enumerate(('BODY','HEAD')):
            old=decisions['OLD'][:,h]==gt['near'][:,h];new=decisions['NEW'][:,h]==gt['near'][:,h]
            paired[name]=dict(old_only_correct=int((old&~new).sum()),new_only_correct=int((~old&new).sum()),
                             both_correct=int((old&new).sum()),both_wrong=int((~old&~new).sum()))
        groups={}
        for group in dict.fromkeys(r['group_id'] for r in rec['records']):
            ids=[i for i,r in enumerate(rec['records']) if r['group_id']==group]
            groups[group]={arm:bool((decisions[arm][ids]==gt['near'][ids]).all()) for arm in decisions}
        paired['groups']=dict(new_only=sum(g['NEW'] and not g['OLD'] for g in groups.values()),
                              old_only=sum(g['OLD'] and not g['NEW'] for g in groups.values()))
        summary[role]['paired']=paired
    write(out/'comparison.json',summary)
    write(out/'validation.json',dict(status='PASS',head_confusions_and_auc=verified,
          checks=['Independent rank-based AUC and inclusive confusion','Result/selection/checkpoint/source/cache receipt hashes','Tied-score low-FP diagnostic'],
          dataset_scope='Shared-asset controlled Development, no natural/general safety claim'))
    print({arm:summary['eval'][arm]['heads'] for arm in ('OLD','NEW')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.cache,a.output)


\n