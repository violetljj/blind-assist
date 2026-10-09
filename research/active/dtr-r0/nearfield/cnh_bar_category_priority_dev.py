"""Aggregate existing event/clear ledgers only; no inference or new experiment."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts.local/work/cnh-bar-category-priority-dev-20261009'
SHAPES = ROOT / 'artifacts.local/work/cnh-aligned-shapes-dev-20261008'
POSE = ROOT / 'artifacts.local/work/cnh-bar-transfer-dev-20261009/pose'
QUERY = ROOT / 'artifacts.local/work/cnh-bar-query-cost-dev-20261009'
GROUPS = ('horizontal_dark4', 'horizontal_other', 'vertical', 'protrusion', 'sign_edge')


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def group(row):
    if row['family'] == 'horizontal':
        return 'horizontal_dark4' if row['rho'] == .25 and 'thick0.04' in row['variant'] else 'horizontal_other'
    return row['family']


def write_csv(path, records):
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    start = time.monotonic()
    inputs = [SHAPES / 'PLAN.json', SHAPES / 'event_ledger.csv', POSE / 'event_ledger.csv',
              POSE / 'result_analysis.json', QUERY / 'clear_slots.csv', QUERY / 'result_diagnosis.json']
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'result.json').exists():
        raise FileExistsError('Preserve completed aggregation')
    rows = json.loads(inputs[0].read_text(encoding='utf-8'))['scene_rows']
    original = read_csv(inputs[1])
    events = read_csv(inputs[2])
    pose_summary = json.loads(inputs[3].read_text(encoding='utf-8'))['summary']
    clear = read_csv(inputs[4])
    query_summary = json.loads(inputs[5].read_text(encoding='utf-8'))['summary']
    assert len(rows) == 492 and len(events) == 26144 and len(clear) == 32032
    assert len({(r['branch'], r['scene'], r['replica'], r['height']) for r in events}) == len(events)
    assert len({(r['angle'], r['scene'], r['replica'], r['frame']) for r in clear}) == len(clear)
    scene_sets = {g: [r['id'] for r in rows if group(r) == g] for g in GROUPS}
    assert sorted(i for ids in scene_sets.values() for i in ids) == list(range(492))
    event_groups = defaultdict(list)
    detail_groups = defaultdict(list)
    for event in events:
        r = rows[int(event['scene'])]
        event_groups[event['branch'], group(r), event['height']].append(event)
        detail_groups[event['branch'], group(r), event['height'], r['placement'], r['context'], r['rho'], r['variant']].append(event)
        assert int(event['gain']) == int(event['M3'] == '0' and event['fusion'] == '1')
        assert int(event['loss']) == int(event['M3'] == '1' and event['fusion'] == '0')

    def counts(ee):
        n = len(ee)
        m = sum(int(e['M3']) for e in ee)
        f = sum(int(e['fusion']) for e in ee)
        gain = sum(int(e['gain']) for e in ee)
        loss = sum(int(e['loss']) for e in ee)
        assert f - m == gain - loss
        return dict(denominator=n, M3=m, fusion=f, M3_miss=n-m, fusion_miss=n-f, gain=gain, loss=loss)

    contact_records = [dict(branch=b['branch'], category=g, height=h, **counts(event_groups[b['branch'], g, h]))
                       for b in pose_summary for g in GROUPS for h in ('HEAD', 'BODY')]
    detail_records = [dict(branch=k[0], category=k[1], height=k[2], placement=k[3], context=k[4], rho=k[5], variant=k[6], **counts(ee))
                      for k, ee in sorted(detail_groups.items())]
    old_timely = {(r['scene'], r['replica'], ('HEAD', 'BODY')[int(r['query'])]): int(r['timely'])
                  for r in original if r['category'] == 'contact'}
    for e in events:
        if e['branch'] == 'ideal':
            assert int(e['M3']) == old_timely[e['scene'], e['replica'], e['height']]
    for b in pose_summary:
        for q, h in enumerate(('HEAD', 'BODY')):
            cc = [r for r in contact_records if r['branch'] == b['branch'] and r['height'] == h]
            assert sum(r['denominator'] for r in cc) == 688
            for method in ('M3', 'fusion'):
                assert sum(r[method] for r in cc) == b[method]['counts'][q]
            for key in ('gain', 'loss'):
                assert sum(r[key] for r in cc) == b['paired'][q][key]

    clear_groups = defaultdict(list)
    for e in clear:
        r = rows[int(e['scene'])]
        assert r['family'] == e['family']
        clear_groups[int(e['angle']), group(r)].append(e)
    clear_records = []
    global_records = []
    for summary in query_summary:
        angle = summary['angle']
        for g in GROUPS:
            ee = clear_groups[angle, g]
            clips = defaultdict(dict)
            for e in ee:
                clips[e['scene'], e['replica']][int(e['frame'])] = e
            assert all(sorted(frames) == list(range(3, 16)) for frames in clips.values())
            rec = dict(angle=angle, category=g, denominator=len(ee), clip_denominator=len(clips))
            for method in ('M3', 'fusion'):
                rec[method+'_slots'] = sum(int(e[method]) for e in ee)
                rec[method+'_segments'] = sum(sum(int(frames[f][method]) and (f == 3 or not int(frames[f-1][method]))
                                                    for f in range(3, 16)) for frames in clips.values())
                rec[method+'_clips'] = sum(any(int(e[method]) for e in frames.values()) for frames in clips.values())
            for key in ('ideal_added', 'ideal_lost'):
                rec[key] = sum(int(e[key]) for e in ee)
            rec['net_vs_ideal'] = rec['ideal_added'] - rec['ideal_lost']
            clear_records.append(rec)
        cc = [r for r in clear_records if r['angle'] == angle]
        assert sum(r['denominator'] for r in cc) == 4576
        assert sum(r['clip_denominator'] for r in cc) == 352
        for method, source in (('M3', 'original'), ('fusion', 'fusion')):
            for metric in ('slots', 'segments', 'clips'):
                assert sum(r[method+'_'+metric] for r in cc) == summary[source]['clear_'+metric]
        global_records.append(dict(angle=angle, denominator=4576, clip_denominator=352,
                                   M3_pass_clips=summary['original']['pass_clips'],
                                   fusion_pass_clips=summary['fusion']['pass_clips'], pass_clip_denominator=352))
    result = dict(status='PASS', scope='Existing consumed Development ledgers, no model/geometry replay or statistics',
                  authorization='User consensus: class ranking from existing ledger; remote ancestry check; no new experiment',
                  budget_CPU_wall_seconds=60, seconds=time.monotonic()-start,
                  inputs_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  scene_sets=scene_sets, event_rows=len(events), clear_slot_rows=len(clear),
                  contact=contact_records, clear=clear_records, shared_global=global_records,
                  checks=['unique event/slot keys', 'disjoint/exhaustive class partition', 'all19 event totals match prior pose summary',
                          'ideal M3 matches original shape event ledger', 'all7 jointclear slot/segment/clip totals match prior query summary'],
                  inference=0, training=0, new_samples=0)
    assert result['seconds'] < 60
    write_csv(OUT / 'contact_by_category.csv', contact_records)
    write_csv(OUT / 'contact_details.csv', detail_records)
    write_csv(OUT / 'query_clear_by_category.csv', clear_records)
    write_csv(OUT / 'shared_pass.csv', global_records)
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], seconds=result['seconds'], contact_rows=len(contact_records),
                          detail_rows=len(detail_records), clear_rows=len(clear_records)), ensure_ascii=False))
    for r in contact_records:
        if r['branch'] == 'ideal':
            print(r)
    for r in clear_records:
        if r['angle'] in (0, 3, -3):
            print(r)


if __name__ == '__main__':
    main()
