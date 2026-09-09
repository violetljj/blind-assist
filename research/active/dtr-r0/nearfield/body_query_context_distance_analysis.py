"""Independent no-inference scoring of frozen context-decoder distance pairs."""
import argparse
from pathlib import Path
import numpy as np
from body_query_data import read, write, sha
from body_query_10000_readout_analysis import TRAINED_10K_SHA, confusion, range_event_from_counts


def analyze(baseline, decoder, distance, run):
    receipt, protocol = read(run/'receipt.json'), read(run/'protocol.json')
    historical = read(distance/'receipt.json')
    assert receipt['status'] == historical['status'] == 'PASS'
    assert receipt['fits'] == receipt['training_steps'] == protocol['fits'] == 0
    assert receipt['baseline_state_unchanged'] and receipt['baseline_prediction_parity'] < 2e-6
    assert protocol['baseline_sha256'] == sha(baseline/'NEW-step2000.pt') == TRAINED_10K_SHA
    assert protocol['source_sha256'] == sha(Path(__file__).with_name('body_query_context_distance.py'))
    assert protocol['distance_receipt_sha256'] == sha(distance/'receipt.json')
    for name, key in [('frame_metadata.json','frame_metadata_sha256'), ('evaluator_truth.npz','evaluator_truth_sha256'), ('predictions.npz','predictions_sha256')]:
        assert sha(distance/name) == historical[key]
    assert sha(run/'predictions.npz') == receipt['predictions_sha256']
    assert sha(run/'result.json') == receipt['result_sha256']
    assert sha(decoder/'normalization.npz') == protocol['normalization_sha256']
    assert sha(decoder/'selection.json') == protocol['selection_sha256'] == read(decoder/'receipt.json')['selection_sha256']
    for arm, digest in protocol['decoder_hashes'].items():
        assert sha(decoder/f'{arm}.pt') == digest == read(decoder/'fits.json')[arm]['sha256']
    rows = read(run/'frame_metadata.json')
    assert rows == read(distance/'frame_metadata.json') and len(rows) == 5000
    assert [row['sample_index'] for row in rows] == list(range(5000))
    # RGB source bytes were hash checked by the inference operator; this audit
    # binds identical metadata to the historical receipt without decoding images.
    with np.load(distance/'evaluator_truth.npz', allow_pickle=False) as gt:
        counts = gt['native_counts'].reshape(-1,12)
        native = gt['native_events'].astype(bool)
    assert np.array_equal(native, counts.reshape(-1,2,2,3).sum(-1) >= 3)
    assert not native[:,0].any() and native[:,1].any(1).all()
    pairs = {}
    for i,row in enumerate(rows):
        pair = pairs.setdefault(row['pair_id'], {})
        assert row['endpoint'] not in pair
        pair[row['endpoint']] = i
    assert len(pairs) == 2500 and all(set(pair) == {'near','far'} for pair in pairs.values())
    for pair in pairs.values():
        n,f = pair['near'], pair['far']
        assert native[n,1].tolist() == [True,False] and native[f,1].tolist() == [False,True]
        assert all(rows[n][key] == rows[f][key] for key in ('source_partition','region_id','site_id','family'))
    arrays = dict(np.load(run/'predictions.npz', allow_pickle=False))
    previous = dict(np.load(distance/'predictions.npz', allow_pickle=False))
    assert set(arrays) == {arm+'_'+key for arm in ('BASE','LOCAL','JOINT') for key in ('counts','near')}
    selection = read(decoder/'selection.json')
    assert selection['BASE']['thresholds'] == read(baseline/'selection.json')['NEW']['thresholds']
    parity = max(float(np.abs(arrays['BASE_'+key]-previous['NEW_'+key]).max()) for key in ('near','counts'))
    assert parity < 2e-6
    threshold = np.asarray(selection['BASE']['thresholds'])
    assert np.array_equal(arrays['BASE_near'] >= threshold, previous['NEW_near'] >= threshold)
    scores = {}
    for arm in ('BASE','LOCAL','JOINT'):
        c,n = arrays[arm+'_counts'], arrays[arm+'_near']
        assert c.shape == (5000,12,4) and n.shape == (5000,2)
        assert np.isfinite(c).all() and np.isfinite(n).all()
        assert ((c>=0)&(c<=1)).all() and ((n>=0)&(n<=1)).all()
        assert np.allclose(c.sum(-1),1,rtol=0,atol=2e-6)
        scores[arm] = range_event_from_counts(c)
    scopes = {'all':np.arange(5000), 'eval':np.array([i for i,row in enumerate(rows) if row['source_partition']=='eval'])}
    for key in ('region_id','family','source_partition'):
        for value in sorted({row[key] for row in rows}):
            label = ('region' if key=='region_id' else key)+':'+value
            scopes[label] = np.array([i for i,row in enumerate(rows) if row[key]==value])
    result = {}
    trained = read(run/'result.json')
    for scope,ids in scopes.items():
        members=set(ids.tolist())
        ps=[pair for pair in pairs.values() if pair['near'] in members and pair['far'] in members]
        assert len(ids) == 2*len(ps)
        result[scope] = {}
        for arm in ('BASE','LOCAL','JOINT'):
            c,n,r=arrays[arm+'_counts'], arrays[arm+'_near'], scores[arm]
            thresholds = np.asarray(selection[arm]['thresholds'])
            delta=np.array([r[p['far'],1,1]-r[p['near'],1,1] for p in ps])
            q=confusion(1-c[ids,6:9,0],counts[ids,6:9]>0,.5)
            value=dict(frames=len(ids),pairs=len(ps),far_higher=int((delta>1e-6).sum()),
                ties=int((np.abs(delta)<=1e-6).sum()),near_higher=int((delta < -1e-6).sum()),
                mean_delta=float(delta.mean()),median_delta=float(np.median(delta)),
                HEAD_near_query=q, HEAD_near_query_recall=q['TP']/(q['TP']+q['FN']),
                HEAD_near_event=confusion(r[ids,1,0],native[ids,1,0],.5),
                HEAD_far_event=confusion(r[ids,1,1],native[ids,1,1],.5),
                exact_HEAD_range=int(((r[ids,1]>=.5)==native[ids,1]).all(1).sum()),
                count_derived_HEAD_hits=int((n[ids,1]>=thresholds[1]).sum()),
                count_derived_BODY_FP=int((n[ids,0]>=thresholds[0]).sum()),
                retained_BASE_HEAD_hits=int((arrays['BASE_near'][ids,1]>=threshold[1]).sum()),
                retained_BASE_BODY_FP=int((arrays['BASE_near'][ids,0]>=threshold[0]).sum()),
                unknown_pixels=int(sum(rows[i]['unknown_pixels'] for i in ids)))
            if scope in trained:
                for key,expected in trained[scope][arm].items():
                    if isinstance(expected,float):
                        assert np.isclose(value[key],expected,rtol=0,atol=1e-12)
                    else:
                        assert value[key] == expected
            result[scope][arm]=value
    assert result['eval']['BASE']['pairs']==750 and result['eval']['BASE']['frames']==1500
    gates={}
    for arm in ('LOCAL','JOINT'):
        v=result['eval'][arm]
        values=dict(near_query_recall_at_least_half=v['HEAD_near_query_recall']>=.5,
                    wrong_far_FP_at_most_150=v['HEAD_far_event']['FP']<=150,
                    exact_HEAD_range_at_least_1200=v['exact_HEAD_range']>=1200)
        gates[arm]=dict(values=values,strong_transfer=all(values.values()))
    summary=dict(status='PASS',scope='Consumed controlled shared-site Development; all HEAD-positive endpoints',
        results=result,gates=gates,baseline_prediction_parity=parity,retained_BASE_decision_parity_exact=True,
        metadata_sha256=sha(run/'frame_metadata.json'),native_truth_sha256=sha(distance/'evaluator_truth.npz'),
        protocol_sha256=sha(run/'protocol.json'),receipt_sha256=sha(run/'receipt.json'),analyzer_sha256=sha(Path(__file__)))
    write(run/'context-distance-analysis.json',summary)
    write(run/'validation.json',dict(status='PASS',fits=0,training_steps=0,no_inference=True,
        analysis_sha256=sha(run/'context-distance-analysis.json'),
        checks=['Receipt/source/checkpoint/normalization/selection bindings','Native and metadata hashes plus native event recomputation',
                'Normalized probabilities, independent paired metrics and slices','Exact retained-BASE alert decisions','Predeclared transfer gates'],
        limits='No HEAD-negative denominator; baseline preservation is a separate two-output interface, not full replacement'))
    print(gates,flush=True)
    print(result['eval'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('baseline','decoder','distance','run'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    analyze(args.baseline,args.decoder,args.distance,args.run)
