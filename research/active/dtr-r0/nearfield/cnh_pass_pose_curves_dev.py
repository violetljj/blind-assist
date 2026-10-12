"""Read-only saved-workpoint cost curves and 1cm HEAD pairing, CPU only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'artifacts.local/work/cnh-pass-boundary-dev-20261009/workpoints_continued'
OUT = ROOT/'artifacts.local/work/cnh-pass-pose-diagnostic-dev-20261009'
POLICIES = ('standalone', 'old_fusion_plus_candidate')
BRANCHES = ('ideal', 'yaw_plus3')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair(before, after):
    if before.keys() != after.keys():
        raise ValueError('Paired 1cm cohort changed')
    gain = sum(not before[k] and after[k] for k in before)
    loss = sum(before[k] and not after[k] for k in before)
    return dict(gain=gain, loss=loss, net=gain-loss, denominator=len(before))


def plot(rows, baseline, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    seeds = sorted({r['seed'] for r in rows})
    colors = {'control':'#47718e', 'weak_pass':'#c87432'}
    markers = ['o','s','^']
    fig, axes = plt.subplots(3,4,figsize=(19,13),constrained_layout=True,
                             gridspec_kw={'height_ratios':[1,1,1.18]})
    for pi, policy in enumerate(POLICIES):
        for bi, branch in enumerate(BRANCHES):
            for ci, (cost,label,den) in enumerate((('clear_clips','Clear clips',512),('pass_clips','Pass clips',256))):
                ax=axes[pi,bi*2+ci]
                for arm in ('control','weak_pass'):
                    for si, seed in enumerate(seeds):
                        rr=[r for r in rows if (r['policy'],r['branch'],r['arm'],r['seed'])==(policy,branch,arm,seed)
                            and r['status']=='CALIBRATED']
                        rr.sort(key=lambda r:r['pass_rate_budget'])
                        ax.plot([r[cost] for r in rr],[r['HEAD'] for r in rr],color=colors[arm],
                                marker=markers[si],lw=1.25,alpha=.85,ms=5)
                for name,marker in (('M3','X'),('old_local_fusion','P')):
                    m=baseline[branch][name]
                    ax.scatter(m[cost],m['counts'][0],marker=marker,color='#303030',s=65,zorder=4)
                ax.set(title=('Single' if pi==0 else 'Old OR candidate')+' / '+branch,
                       xlabel=f'{label} / {den}',ylabel='HEAD timely / 384',ylim=(170,365))
                ax.grid(alpha=.2)
    def heat(ax, entries, title, labels):
        values=np.full((len(entries),3),np.nan)
        for i,rr in enumerate(entries):
            for j,r in enumerate(rr):
                if r is not None:
                    values[i,j]=r['gain']-r['loss']
        cm=plt.get_cmap('RdBu').copy();cm.set_bad('#e3e3e3')
        ax.imshow(np.ma.masked_invalid(values),cmap=cm,vmin=-30,vmax=30,aspect='auto')
        for i,rr in enumerate(entries):
            for j,r in enumerate(rr):
                ax.text(j,i,'N/P' if r is None else f"{r['gain']}/{r['loss']}",ha='center',va='center',fontsize=9)
        ax.set(xticks=[0,1,2],xticklabels=['20%','30%','40%'],yticks=list(range(len(labels))),
               yticklabels=labels,title=title,xlabel='Frozen ideal-cal pass budget; cells = gain/loss')
        ax.tick_params(axis='y',labelsize=8)
    lookup={(r['arm'],r['seed'],r['policy'],r['point'],r['branch']):r for r in rows}
    for pi,policy in enumerate(POLICIES):
        entries,labels=[],[]
        for branch in BRANCHES:
            for seed in seeds:
                rr=[]
                for p in ('pass_20pct','pass_30pct','pass_40pct'):
                    r=lookup['weak_pass',seed,policy,p,branch]
                    rr.append(None if r['status']!='CALIBRATED' else dict(
                        gain=r['weak_vs_control_1cm_HEAD_gain'],loss=r['weak_vs_control_1cm_HEAD_loss']))
                entries.append(rr);labels.append(('Ideal' if branch=='ideal' else '+3')+' '+str(seed)[-2:])
        heat(axes[2,pi],entries,('Single' if pi==0 else 'OR')+' 1cm HEAD: weak minus control',labels)
        entries,labels=[],[]
        for arm in ('control','weak_pass'):
            for seed in seeds:
                rr=[]
                for p in ('pass_20pct','pass_30pct','pass_40pct'):
                    r=lookup[arm,seed,policy,p,'yaw_plus3']
                    rr.append(None if r['status']!='CALIBRATED' else dict(
                        gain=r['ideal_to_yaw_1cm_HEAD_gain'],loss=r['ideal_to_yaw_1cm_HEAD_loss']))
                entries.append(rr);labels.append(('Ctrl' if arm=='control' else 'Weak')+' '+str(seed)[-2:])
        heat(axes[2,pi+2],entries,('Single' if pi==0 else 'OR')+' 1cm HEAD: ideal to +3',labels)
    handles=[Line2D([0],[0],color=c,label=a) for a,c in colors.items()]
    handles += [Line2D([0],[0],color='#555555',marker=m,ls='',label=str(s)) for s,m in zip(seeds,markers)]
    handles += [Line2D([0],[0],color='#303030',marker=m,ls='',label=n) for n,m in [('M3','X'),('Old 5-slot','P')]]
    fig.suptitle('Saved validation workpoints: clear clips and pass clips remain separate\n'
                 'Connected points are cal budgets 20 -> 30 -> 40%; correlated simulated fixtures, no retuning',fontsize=14)
    fig.legend(handles=handles,loc='outside lower center',ncol=7,fontsize=10)
    fig.savefig(path,dpi=170);plt.close(fig)


def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']
                     +['| '+' | '.join(map(str,r))+' |' for r in rows])


def run(source=SOURCE, output=OUT):
    start=time.monotonic();source,output=Path(source).resolve(),Path(output).resolve()
    plan=read(output/'PLAN.json');cap=float(plan['allocations']['curves'])
    if cap>70:
        raise ValueError('This diagnostic allocation cannot exceed the authorized 70 seconds')
    names=('curves.csv','curves.png','curves_result.json','curves_receipt.json','curves_findings.md')
    if any((output/n).exists() for n in names):
        raise FileExistsError('Preserve existing curves outputs')
    receipt=dict(status='RUNNING',cpu_only=True,training=0,inference=0,new_thresholds=0,
                 source_sha256=sha(__file__),plan_sha256=sha(output/'PLAN.json'))
    def check():
        if time.monotonic()-start>=cap:
            raise TimeoutError('Curves CPU stage allocation reached')
    try:
        thresholds=read(source/'thresholds.json')['arms']
        metrics=read(source/'metrics.json')
        lookup={}
        with (source/'summary.csv').open(encoding='utf8',newline='') as handle:
            for r in csv.DictReader(handle):
                if r['dataset'].startswith('validation/') and r['arm'] in plan['arms'] and r['point'] in plan['points']:
                    key=(r['arm'],int(r['seed']),r['policy'],r['point'],r['dataset'].split('/')[1],r['stratum'])
                    if key in lookup:raise ValueError('Duplicate saved summary identity')
                    lookup[key]=r
        events={};baseline_events={b:{} for b in BRANCHES}
        with (source/'event_ledger.csv').open(encoding='utf8',newline='') as handle:
            for r in csv.DictReader(handle):
                if not (r['dataset'].startswith('validation/') and r['arm'] in plan['arms']
                        and r['point'] in plan['points'] and r['placement']=='in1cm' and r['height']=='HEAD'):
                    continue
                branch=r['dataset'].split('/')[1];key=(r['arm'],int(r['seed']),r['policy'],r['point'],branch)
                event=(int(r['scene']),int(r['replica']))
                by_event=events.setdefault(key,{})
                if event in by_event:raise ValueError('Duplicate saved event identity')
                by_event[event]=int(r['timely'])
                base=tuple(int(r[k]) for k in ('M3_timely','old_local_fusion_timely'))
                if event in baseline_events[branch] and baseline_events[branch][event]!=base:
                    raise ValueError('Fixed baseline changed across workpoint ledger rows')
                baseline_events[branch][event]=base
        check()
        baselines={b:metrics['datasets']['validation/'+b]['baseline'] for b in BRANCHES}
        for branch in BRANCHES:
            if len(baseline_events[branch])!=128:raise ValueError('Expected 128 one-cm HEAD baseline events')
            for q,name in enumerate(('M3','old_local_fusion')):
                baselines[branch][name]['HEAD_1cm_count']=sum(v[q] for v in baseline_events[branch].values())
                baselines[branch][name]['HEAD_1cm_denominator']=128
        rows=[]
        for arm in plan['arms']:
            for seed in plan['seeds']:
                for policy in POLICIES:
                    for point in plan['points']:
                        threshold=thresholds[arm][str(seed)][policy][point]
                        for branch in BRANCHES:
                            key=(arm,seed,policy,point,branch)
                            r=dict(arm=arm,seed=seed,policy=policy,point=point,branch=branch,
                                   status=threshold['status'],pass_rate_budget=threshold['pass_rate_budget'],
                                   cal_pass_cap=threshold['pass_cap'],cal_clear_cap=threshold['clear_cap'],
                                   threshold=threshold['threshold'],clear_slots=None,clear_slot_denominator=6656,
                                   clear_segments=None,clear_clips=None,clear_clip_denominator=512,
                                   pass_clips=None,pass_clip_denominator=256,HEAD=None,BODY=None,
                                   HEAD_denominator=384,BODY_denominator=384,HEAD_1cm=None,HEAD_1cm_denominator=128)
                            pair_fields=['weak_vs_control_1cm_HEAD_gain','weak_vs_control_1cm_HEAD_loss',
                                         'ideal_to_yaw_1cm_HEAD_gain','ideal_to_yaw_1cm_HEAD_loss']
                            pair_fields += [f'{height}_{field}_vs_{base}{suffix}' for suffix in ('','_1cm')
                                            for base in ('M3','old_local_fusion','control')
                                            for height in ('HEAD','BODY') for field in ('gain','loss','net')]
                            r.update({name:None for name in pair_fields})
                            if r['status']=='CALIBRATED':
                                overall=lookup[(*key,'all')];one=lookup[(*key,'placement:in1cm')]
                                for field in ('HEAD','BODY','clear_slots','clear_segments','clear_clips','pass_clips'):
                                    r[field]=int(overall[field])
                                r['HEAD_1cm']=int(one['HEAD'])
                                before=events[arm,seed,policy,point,'ideal'];after=events[arm,seed,policy,point,'yaw_plus3']
                                transport=pair(before,after)
                                if transport['denominator']!=128:raise ValueError('Changed one-cm denominator')
                                r.update(ideal_to_yaw_1cm_HEAD_gain=transport['gain'],ideal_to_yaw_1cm_HEAD_loss=transport['loss'])
                                wc=pair(events['control',seed,policy,point,branch],events['weak_pass',seed,policy,point,branch])
                                r.update(weak_vs_control_1cm_HEAD_gain=wc['gain'],weak_vs_control_1cm_HEAD_loss=wc['loss'])
                                for suffix,rr in (('',overall),('_1cm',one)):
                                    for base,source_key in (('M3','M3'),('old_local_fusion','old_local_fusion'),
                                                            ('control','control_same_point_same_policy')):
                                        for height in ('HEAD','BODY'):
                                            for field in ('gain','loss','net'):
                                                r[f'{height}_{field}_vs_{base}{suffix}']=int(rr[f'{height}_{field}_vs_{source_key}'])
                                weak=lookup[('weak_pass',seed,policy,point,branch,'placement:in1cm')]
                                if wc['gain']!=int(weak['HEAD_gain_vs_control_same_point_same_policy']) or wc['loss']!=int(weak['HEAD_loss_vs_control_same_point_same_policy']):
                                    raise ValueError('Ledger weak-control pair differs from saved summary')
                            rows.append(r)
        check()
        fields=[k for k in rows[0] if k!='threshold']+['threshold_value','threshold_positive_infinity']
        with (output/'curves.csv').open('x',encoding='utf8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
            for r in rows:
                rr={k:v for k,v in r.items() if k!='threshold'}
                rr.update(threshold_value=r['threshold']['value'] if r['threshold'] else None,
                          threshold_positive_infinity=r['threshold']['positive_infinity'] if r['threshold'] else None)
                writer.writerow(rr)
        plot(rows,baselines,output/'curves.png');check()
        result=dict(status='COMPLETE',rows=rows,baselines=baselines,
                    pairing_key='same validation scene_id x replica x HEAD; exact same policy and cal pass point',
                    all_points_retained=True,threshold_selection=0,training=0,inference=0,
                    exposure='HEAD/BODY each384; jointclear512 clips and6656 slots; pass256clips; onecm HEAD128',
                    limits='Consumed simulated Development; correlated replicas/templates/seeds. No -3 branch. No deployment point selection or zero-loss gate.')
        write(output/'curves_result.json',result)
        lines=['# 已有工作点的双条件 clip 成本曲线', '',
               '完整保留 control/weak_pass ×三个 seed×20/30/40% 校准预算×ideal/+3°。'
               '图的 clear clip/512 与 pass clip/256 分开；slots/6656 独立保存在 CSV，不跨分母相抵。'
               '所有阈值沿用旧 ideal-cal 选择，没有新搜索、训练、推理或上线点选择。', '',
               '40% 独立读出展示总体优势与浅侵入限制：在 +3° 上 weak 的总体 HEAD/BODY 均高于冻结 M3，'
               'clear clips 和 pass clips 均较 M3 少；ideal 上也有总体命中收益，但 pass 高于 M3。'
               '这不应被某个1cm损失或零损失门槛抹去，也不代表所有条件都同成本支配 M3/旧融合。', '']
        chosen=[r for r in rows if r['arm']=='weak_pass' and r['policy']=='standalone' and r['point']=='pass_40pct']
        tab=[]
        for r in chosen:
            tab.append([r['seed'],r['branch'],r['HEAD'],r['BODY'],f"{r['clear_clips']}/512",f"{r['pass_clips']}/256",
                        f"{r['clear_slots']}/6656",f"{r['HEAD_1cm']}/128",
                        f"{r['HEAD_gain_vs_M3_1cm']}/{r['HEAD_loss_vs_M3_1cm']}",
                        f"{r['HEAD_gain_vs_old_local_fusion_1cm']}/{r['HEAD_loss_vs_old_local_fusion_1cm']}",
                        f"{r['weak_vs_control_1cm_HEAD_gain']}/{r['weak_vs_control_1cm_HEAD_loss']}",
                        f"{r['ideal_to_yaw_1cm_HEAD_gain']}/{r['ideal_to_yaw_1cm_HEAD_loss']}"])
        lines += [mdtable(['seed','条件','HEAD/384','BODY/384','clearclip','passclip','clearslot','1cmHEAD',
                          '1cm vsM3救/损','1cm vsOld救/损','1cm weak-control救/损','该model ideal→yaw救/损'],tab),'',
                  '40% 是已声明曲线上的一个展示例；20/30/40% 全部结果见 curves.csv 和图，未因验证表现重选部署政策。', '',
                  '基线实际成本与命中：','',mdtable(['条件','参照','HEAD/384','BODY/384','clearclip/512','passclip/256','slot/6656','1cmHEAD/128'],
                      [[b,n,m['counts'][0],m['counts'][1],m['clear_clips'],m['pass_clips'],m['clear_slots'],m['HEAD_1cm_count']]
                       for b,bb in baselines.items() for n,m in bb.items()]), '',
                  '图中底部灰格 N/P 为固定旧 OR pass floor55超过20% cal预算51；没有撤回旧报警。'
                  '底部所有救/损以1cm HEAD/128匹配事件计算，未把“miss”过滤掉。'
                  'ideal→yaw 的列为同一 model 固定阈值下的事件变化；不能把不同 model 的 logit 尺度差当作姿态效应。', '',
                  '两个新增背景族、各两实例和固定 shape-grid，均属已消费模拟 Development；相关重复不是 iid 世界数。'
                  '没有−3°分支，因此不推断扰动符号对称性、实机坐标准确性或安全效果。']
        (output/'curves_findings.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
        receipt.update(status='COMPLETE',curve_rows=len(rows),feasible_rows=sum(r['status']=='CALIBRATED' for r in rows),
                       onecm_join_checks='PASS; every feasible model/point has128same-IDHEAD events',
                       input_root=str(source),input_hashes='Root existing input binding; no duplicate full-payload hashing',
                       outputs={n:sha(output/n) for n in names if n!='curves_receipt.json'})
    except Exception as error:
        receipt.update(status='FAILED',error=repr(error));raise
    finally:
        receipt.update(seconds=time.monotonic()-start,allocated_cpu_seconds=cap,total_task_cpu_cap_seconds=plan['cpu_analysis_cap_seconds'])
        write(output/'curves_receipt.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();run(args.source,args.output)
