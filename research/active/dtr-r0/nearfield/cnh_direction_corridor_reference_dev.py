"""Numpy reference for a past-displacement direction and future corridor proxy.

Positions must be in meters. Horizontal axes are explicit input, never inferred
from field names. Anchors are actual native frames: first frame at/after each
one-second segment-relative target. No future sample interpolates the anchor
or contributes to the past direction. Future trajectories are evaluation only.

The future horizontal polyline plus a .30m disk is a center-trajectory swept
proxy, not a measured body or a predicted walking corridor. Maximum center
distance from the endpoint chord is not a body-region omission proportion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


OUT = Path(__file__).resolve().parents[4]/'artifacts.local/work/cnh-direction-contract-dev-20261009'
HORIZONS_S = (.5, 1., 1.5)
EXPECTED_PERIOD_NS = 1e9/240
INTERVAL_TOLERANCE_NS = 1_000_000


def horizontal_basis(axes):
    """Two distinct coordinate indices, or explicit orthonormal basis[2,3]."""
    axes=np.asarray(axes)
    if axes.shape==(2,):
        if not np.issubdtype(axes.dtype,np.integer) or len(set(axes.tolist()))!=2 or not np.isin(axes,[0,1,2]).all():
            raise ValueError('Explicit two distinct horizontal coordinate indices required')
        return np.eye(3)[axes.astype(int)]
    if axes.shape!=(2,3) or not np.isfinite(axes).all() or not np.allclose(axes@axes.T,np.eye(2),atol=1e-10,rtol=0):
        raise ValueError('Explicit horizontal basis must be orthonormal [2,3]')
    return axes.astype(np.float64)


def legal_segments(timestamps_ns,split_indices=None,*,sample_hz=240.):
    """Return [begin,end) native row segments and invalid edge diagnostics.

    Optional split_indices are native segment starts; 0/N boundaries are also
    accepted. Nonpositive or >1ms-deviant intervals from explicit sample_hz
    independently force cuts. The sampling rate must be positive and finite.
    Every row belongs to one segment; no illegal edge is silently bridged.
    """
    try:sample_hz=float(sample_hz)
    except (TypeError,ValueError,OverflowError) as error:
        raise ValueError('Positive finite sample_hz required') from error
    if not np.isfinite(sample_hz) or sample_hz<=0:raise ValueError('Positive finite sample_hz required')
    t=np.asarray(timestamps_ns)
    if t.ndim!=1 or not np.issubdtype(t.dtype,np.integer):
        raise ValueError('Integral native timestamps_ns[N] required')
    n=len(t)
    if not n:return np.empty((0,2),np.int64),dict(edge_after_index=np.empty(0,np.int64),interval_ns=np.empty(0,np.float64),reason=np.empty(0,'U32'))
    dt=np.diff(t.astype(np.float64))
    bad=(dt<=0)|(np.abs(dt-1e9/sample_hz)>INTERVAL_TOLERANCE_NS)
    cuts=set((np.flatnonzero(bad)+1).tolist())
    if split_indices is not None:
        split=np.asarray(split_indices)
        if split.ndim!=1 or not np.issubdtype(split.dtype,np.integer) or not ((split>=0)&(split<=n)).all():
            raise ValueError('split_indices are native starts in 0..N')
        cuts.update(split.astype(int).tolist())
    boundaries=np.array(sorted(cuts|{0,n}),np.int64)
    segments=np.stack((boundaries[:-1],boundaries[1:]),-1)
    ix=np.flatnonzero(bad)
    reasons=np.where(dt[ix]<=0,'NON_MONOTONIC','INTERVAL_DEVIATION_GT_1MS')
    return segments,dict(edge_after_index=ix.astype(np.int64),interval_ns=dt[ix],reason=reasons)


def _interpolate(t,p,target,begin,end):
    """Inclusive begin, exclusive end bounds; caller fixes estimator cutoff."""
    if target<t[begin] or target>t[end-1]:return None
    right=int(np.searchsorted(t[begin:end],target,side='left'))+begin
    if right<end and t[right]==target:return p[right].copy()
    if right==begin or right>=end:return None
    left=right-1
    weight=(target-int(t[left]))/(int(t[right])-int(t[left]))
    return p[left]*(1-weight)+p[right]*weight


def _max_chord_distance(path):
    vector=path[-1]-path[0];squared=float(vector@vector)
    if squared==0:return float(np.linalg.norm(path-path[0],axis=-1).max())
    ratio=np.clip((path-path[0])@vector/squared,0,1)
    return float(np.linalg.norm(path-(path[0]+ratio[:,None]*vector),axis=-1).max())


def corridor_reference(timestamps_ns,positions,*,horizontal_axes,split_indices=None,min_displacement_m=.02,sample_hz=240.):
    """Full anchor x {.5,1,1.5}s table with unavailable/low-motion rows.

    Columns are [R] arrays. Ragged paths have [R+1] offsets, native/interpolated
    timestamps and full3D/horizontal positions. Missing future windows have an
    empty path. Past direction is anchor minus interpolated anchor-1s, whose
    interpolation bracket never exceeds the actual anchor native frame.
    Directions use atan2(second explicit axis, first explicit axis); wrapped
    error is future chord angle minus past proxy angle in [-pi,pi].
    """
    t=np.asarray(timestamps_ns)
    p=np.asarray(positions,np.float64)
    if p.shape!=(len(t),3) or not np.isfinite(p).all():raise ValueError('Finite metric positions[N,3] required')
    if not np.isfinite(min_displacement_m) or min_displacement_m<=0:raise ValueError('Positive finite numerical direction floor required')
    basis=horizontal_basis(horizontal_axes)
    segments,edges=legal_segments(t,split_indices,sample_hz=sample_hz)
    xy=p@basis.T
    rows=[];paths=[];path3=[];path_times=[];offsets=[0]
    for segment_id,(begin,end) in enumerate(segments):
        begin,end=int(begin),int(end)
        targets=range(int(t[begin]),int(t[end-1])+1,1_000_000_000)
        seen=set()
        for target in targets:
            anchor=int(np.searchsorted(t[begin:end],target,side='left'))+begin
            if anchor>=end or anchor in seen:continue
            seen.add(anchor)
            anchor_t=int(t[anchor]);past_target=anchor_t-1_000_000_000
            # Only rows up to the actual anchor are accessible to the estimator.
            past=_interpolate(t,p,past_target,begin,anchor+1)
            past_length=np.nan;past_angle=np.nan
            if past is None:past_status='NOT_AVAILABLE_PAST_WINDOW'
            else:
                displacement=(p[anchor]-past)@basis.T
                past_length=float(np.linalg.norm(displacement))
                past_status='LOW_MOTION_PAST' if past_length<min_displacement_m else 'AVAILABLE'
                if past_status=='AVAILABLE':past_angle=float(np.arctan2(displacement[1],displacement[0]))
            for horizon in HORIZONS_S:
                future_t=anchor_t+int(round(horizon*1e9))
                endpoint=_interpolate(t,p,future_t,anchor,end)
                chord=path_length=maximum=future_angle=error=np.nan
                if endpoint is None:
                    future_status='NOT_AVAILABLE_FUTURE_WINDOW';future_direction_status=future_status
                else:
                    inside=np.flatnonzero((t[anchor:end]>anchor_t)&(t[anchor:end]<future_t))+anchor
                    times=np.concatenate(([anchor_t],t[inside],[future_t])).astype(np.int64)
                    xyz=np.concatenate((p[anchor:anchor+1],p[inside],endpoint[None]))
                    horizontal=xyz@basis.T
                    vector=horizontal[-1]-horizontal[0]
                    chord=float(np.linalg.norm(vector))
                    path_length=float(np.linalg.norm(np.diff(horizontal,axis=0),axis=-1).sum())
                    maximum=_max_chord_distance(horizontal)
                    future_status='AVAILABLE'
                    future_direction_status='LOW_MOTION_FUTURE_CHORD' if chord<min_displacement_m else 'AVAILABLE'
                    if future_direction_status=='AVAILABLE':future_angle=float(np.arctan2(vector[1],vector[0]))
                    paths.append(horizontal);path3.append(xyz);path_times.append(times)
                    if past_status=='AVAILABLE' and future_direction_status=='AVAILABLE':
                        difference=future_angle-past_angle
                        error=float(np.arctan2(np.sin(difference),np.cos(difference)))
                offsets.append(offsets[-1]+(len(horizontal) if endpoint is not None else 0))
                if past_status!='AVAILABLE':status=past_status
                elif future_status!='AVAILABLE':status=future_status
                elif future_direction_status!='AVAILABLE':status=future_direction_status
                else:status='AVAILABLE'
                rows.append(dict(segment_id=segment_id,segment_begin_index=begin,segment_end_exclusive=end,
                    anchor_index=anchor,anchor_target_timestamp_ns=target,anchor_timestamp_ns=anchor_t,
                    anchor_target_delay_ns=anchor_t-target,horizon_s=horizon,past_window_begin_ns=past_target,
                    future_window_end_ns=future_t,past_status=past_status,future_status=future_status,
                    future_direction_status=future_direction_status,status=status,
                    past_displacement_m=past_length,past_direction_rad=past_angle,future_chord_direction_rad=future_angle,
                    wrapped_error_rad=error,wrapped_error_deg=float(np.degrees(error)),path_length_m=path_length,
                    chord_length_m=chord,max_future_center_distance_to_chord_m=maximum))
    names=('segment_id','segment_begin_index','segment_end_exclusive','anchor_index','anchor_target_timestamp_ns','anchor_timestamp_ns',
           'anchor_target_delay_ns','horizon_s','past_window_begin_ns','future_window_end_ns','past_status','future_status',
           'future_direction_status','status','past_displacement_m','past_direction_rad','future_chord_direction_rad',
           'wrapped_error_rad','wrapped_error_deg','path_length_m','chord_length_m','max_future_center_distance_to_chord_m')
    integer=set(names[:7])|{'past_window_begin_ns','future_window_end_ns'}
    strings={'past_status','future_status','future_direction_status','status'}
    columns={name:np.array([row[name] for row in rows],dtype=np.int64 if name in integer else 'U40' if name in strings else np.float64) for name in names}
    return dict(columns=columns,segments=segments,invalid_edges=edges,
                future_path_offsets=np.array(offsets,np.int64),
                future_path_xy=np.concatenate(paths) if paths else np.empty((0,2)),
                future_path_positions=np.concatenate(path3) if path3 else np.empty((0,3)),
                future_path_timestamps_ns=np.concatenate(path_times) if path_times else np.empty(0,np.int64),
                configuration=dict(horizontal_basis=basis.tolist(),position_unit='meters',timestamp_unit='nanoseconds',
                    horizons_s=list(HORIZONS_S),anchor_target_interval_s=1.,past_window_s=1.,
                    expected_sample_hz=float(sample_hz),interval_tolerance_ns=INTERVAL_TOLERANCE_NS,
                    numerical_direction_floor_m=float(min_displacement_m),direction_floor_scope='numerical proxy validity, not walking threshold',
                    swept_proxy_radius_m=.30,swept_proxy='complete future horizontal center polyline Minkowski-summed with radius.30m disk; not true body',
                    chord_deviation_scope='max center distance to endpoint chord segment; not body-region omission proportion',
                    estimator_scope='past1s displacement only; native anchor; no cross-gap, extrapolation or future estimator samples'))


def focused_fixtures():
    started=time.monotonic();checks=[]
    times=np.rint(np.arange(721)*1e9/240).astype(np.int64);seconds=times/1e9
    pos=np.column_stack((.4*seconds,np.zeros((len(times),2))))
    result=corridor_reference(times,pos,horizontal_axes=(0,1));c=result['columns']
    keep=c['status']=='AVAILABLE'
    np.testing.assert_allclose(c['wrapped_error_rad'][keep],0,atol=1e-12)
    np.testing.assert_allclose(c['path_length_m'][keep],.4*c['horizon_s'][keep],atol=1e-12)
    assert np.all(c['max_future_center_distance_to_chord_m'][keep]<1e-12)
    # Native anchor, not a location interpolated using a frame after the anchor.
    shifted=times.copy();shifted[240:]+=100
    native=corridor_reference(shifted,pos,horizontal_axes=(0,1))['columns']
    assert np.all(native['anchor_timestamp_ns']==shifted[native['anchor_index']])
    checks.append('straight: exact direction/length and native anchors')
    turn=np.column_stack((np.minimum(2*seconds,3),np.maximum(2*seconds-3,0),np.zeros(len(times))))
    turned=corridor_reference(times,turn,horizontal_axes=(0,1));cc=turned['columns']
    row=int(np.flatnonzero((cc['anchor_timestamp_ns']==1_000_000_000)&(cc['horizon_s']==1))[0])
    assert abs(cc['path_length_m'][row]-2)<1e-12 and abs(cc['chord_length_m'][row]-np.sqrt(2))<1e-12
    assert abs(cc['wrapped_error_deg'][row]-45)<1e-10
    assert abs(cc['max_future_center_distance_to_chord_m'][row]-1/np.sqrt(2))<1e-12
    lo,hi=turned['future_path_offsets'][row:row+2];assert hi-lo==241
    checks.append('L turn: complete ragged path and chord deviation differ')
    mask=(times<=1_500_000_000)|(times>=2_000_000_000)
    gap=corridor_reference(times[mask],pos[mask],horizontal_axes=(0,1));gc=gap['columns']
    assert len(gap['segments'])==2 and len(gap['invalid_edges']['edge_after_index'])==1
    assert ((gc['anchor_timestamp_ns']==1_000_000_000)&(gc['horizon_s']==1)&(gc['future_status']=='NOT_AVAILABLE_FUTURE_WINDOW')).any()
    assert (gc['past_status'][gc['anchor_timestamp_ns']==2_000_000_000]=='NOT_AVAILABLE_PAST_WINDOW').all()
    explicit,_=legal_segments(times,split_indices=[240]);assert len(explicit)==2
    duplicate=times.copy();duplicate[240]=duplicate[239];assert len(legal_segments(duplicate)[0])==3
    checks.append('gap: all segments kept, explicit/illegal cuts and no bridged windows')
    boundary=corridor_reference(times[:481],pos[:481],horizontal_axes=(0,1));bc=boundary['columns']
    exact=(bc['anchor_timestamp_ns']==1_000_000_000)&(bc['horizon_s']==1)
    assert (bc['future_status'][exact]=='AVAILABLE').all()
    last=bc['anchor_timestamp_ns']==2_000_000_000
    assert (bc['future_status'][last]=='NOT_AVAILABLE_FUTURE_WINDOW').all()
    assert len(bc['status'])==9 and (bc['past_status'][bc['anchor_index']==0]=='NOT_AVAILABLE_PAST_WINDOW').all()
    checks.append('end boundary: exact endpoint accepted; censored rows retained')
    still=corridor_reference(times[:481],np.zeros((481,3)),horizontal_axes=(0,1));sc=still['columns']
    midpoint=(sc['anchor_timestamp_ns']==1_000_000_000)&(sc['horizon_s']==.5)
    assert (sc['past_status'][midpoint]=='LOW_MOTION_PAST').all()
    assert (sc['future_direction_status'][midpoint]=='LOW_MOTION_FUTURE_CHORD').all()
    assert np.isnan(sc['wrapped_error_rad'][midpoint]).all() and (sc['path_length_m'][midpoint]==0).all()
    checks.append('low motion: separate availability/direction statuses; no invented direction')
    # 60Hz shares the same native anchor/causal-window contract, not 240Hz cuts.
    times60=np.rint(np.arange(181)*1e9/60).astype(np.int64)
    times60[60:]+=100
    positions60=np.column_stack((.4*times60/1e9,np.zeros((len(times60),2))))
    sixty=corridor_reference(times60,positions60,horizontal_axes=(0,1),sample_hz=60.)
    s60=sixty['columns']
    assert len(sixty['segments'])==1 and sixty['configuration']['expected_sample_hz']==60.
    np.testing.assert_array_equal(s60['anchor_timestamp_ns'],times60[s60['anchor_index']])
    assert (s60['anchor_target_delay_ns'][s60['anchor_index']==60]==100).all()
    assert (s60['future_status'][s60['anchor_index']==180]=='NOT_AVAILABLE_FUTURE_WINDOW').all()
    retained=np.arange(len(times60))!=90
    dropped=corridor_reference(times60[retained],positions60[retained],horizontal_axes=(0,1),sample_hz=60.)
    assert len(dropped['segments'])==2 and len(dropped['invalid_edges']['edge_after_index'])==1
    assert (dropped['columns']['future_status'][dropped['columns']['anchor_index']==60]=='NOT_AVAILABLE_FUTURE_WINDOW').all()
    changed=positions60.copy();changed[61:,1]=50.
    future_change=corridor_reference(times60,changed,horizontal_axes=(0,1),sample_hz=60.)['columns']
    anchor60=s60['anchor_index']==60
    np.testing.assert_array_equal(future_change['past_direction_rad'][anchor60],s60['past_direction_rad'][anchor60])
    assert (future_change['future_chord_direction_rad'][anchor60]!=s60['future_chord_direction_rad'][anchor60]).all()
    for invalid_hz in (0.,-1.,np.inf,np.nan):
        try:legal_segments(times60,sample_hz=invalid_hz)
        except ValueError:pass
        else:raise AssertionError('Invalid sample_hz accepted')
    checks.append('60Hz: native anchors, tail censor, dropped-frame gap, future-independent past direction and rate validation')
    return dict(status='PASS',checks=checks,seconds=time.monotonic()-started,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),real_data_opened=False,
                fixtures=6,native_anchor=True,training=0,inference=0,gpu=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',required=True)
    parser.add_argument('--output',type=Path,default=OUT/'focused_reference_check.json')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing fixture receipt')
    started=time.monotonic()
    try:receipt=focused_fixtures()
    except Exception as error:
        receipt=dict(status='FAILED',error=repr(error),seconds=time.monotonic()-started,real_data_opened=False)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8');raise
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt))
