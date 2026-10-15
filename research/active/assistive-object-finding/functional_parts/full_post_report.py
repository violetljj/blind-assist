"""Final full-run evidence index; different supervision/metric families stay separate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time


def read(path):
    p=Path(path)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=root/'final';out.mkdir(exist_ok=True)
    shared=[1,2,5,7,8,9]
    models={}
    for name in ['segformer','dino']:
        native=read(root/'analysis'/name/'summary.json')
        transfer=read(root/'transfer'/name/'evaluation'/'summary.json')
        overall=(native or {}).get('overall',{})
        ious=overall.get('per_class_IoU',{})
        values=[ious[str(c)] for c in shared if ious.get(str(c)) is not None]
        models[name]={'native':native,'UMD_transfer':transfer,
                      'native_mIoU_same_shared6_classes':sum(values)/len(values) if values else None,
                      'training':read(root/'runs'/name/'terminal.json'),
                      'queue_training':read(root/'runs'/name/'queue_terminal.json')}
    evidence={'created_unix':time.time(),'models':models,
              'PLSP_training':read(root/'plsp/train_seed10000/terminal.json'),
              'PLSP_Seen_test':read(root/'plsp/final_seen_test/metrics.json'),
              'UMD_census':read(root/'data/umd/manual_index_census.json'),
              'IIT_manifest':read(root/'metadata/iit_training_manifest_receipt.json'),
              'scope':'Full native-label baseline study. Semantic class/part labels are not free-task role or physical interaction success.'}
    (out/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 全量功能区域基线与独立来源评估','',
           '本文件只汇总已存在结果；缺失项目明确为未完成，不能将排队/启动认定为成果。','',
           '| 模型 | 训练终态 | 原生验证图数 | 原生9功能类mIoU | 同6共享类原生mIoU | UMD共享6类mIoU |',
           '| --- | --- | ---: | ---: | ---: | ---: |']
    def pct(v):return f'{100*v:.3f}%' if isinstance(v,(int,float)) else 'NOT_RUN'
    for name,d in models.items():
        n=(d['native'] or {}).get('overall',{});u=d['UMD_transfer'] or {}
        uvalue=u.get('shared6_foreground_mIoU_primary')
        if isinstance(uvalue,dict):uvalue=uvalue.get('mIoU_foreground')
        lines.append(f"| {name} | {(d['training'] or {}).get('status','NOT_RUN')} | {n.get('images','?')} | {pct(n.get('mIoU_foreground'))} | {pct(d['native_mIoU_same_shared6_classes'])} | {pct(uvalue)} |")
    lines+=['','不同来源包含不同对象、尺寸、标注规则，以上两列不是隔离域因素的因果对照。',
            '原生9类与共享6类也不可直接混排；PLSP的KLD/SIM/NSS在单独metrics.json报告。',
            '完整每类、点命中、同图正类联合成功、混淆、区域大小、来源分组CI和自动失败图见analysis/与transfer/。',
            '角色可研究性以metadata/ROLE_FEASIBILITY.md为准：PACO有部位层级，IIT有语义功能区，均不能自动变成任务角色真值。',
            '训练曲线和最后阶段改进幅度见summary/REPORT.md、analysis/curves/；跑满80epoch本身不证明收敛。',
            'PLSP完整官方test包含同图多动作和历史消费，fresh_without_train_overlap单列，不能把1710行当1710独立新图。',
            'UMD按作者代码选择手标帧，scoop无IIT对应置ignore255，不作为背景；预测不按GT屏蔽或重归一化。',
            '当前结果仅为离线语义区域定位，不包含真实视障操作、安全抓握或指定实例正确性验证。','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    complete=all((d['training'] or {}).get('status')=='COMPLETE' and d['native'] and d['UMD_transfer'] for d in models.values())
    complete=bool(complete and evidence['PLSP_Seen_test'] and evidence['UMD_census'])
    (out/'terminal.json').write_text(json.dumps({'status':'COMPLETE' if complete else 'INCOMPLETE','evidence':str(out/'evidence.json')},indent=2),encoding='utf-8')
    print(json.dumps({'report':str(out/'REPORT.md')}))
    if not complete:raise SystemExit('Required full-run evidence still missing')


if __name__=='__main__':main()
