"""Descriptive sensitivity only; evaluator labels are never model inputs."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[4]/'artifacts.local/work/assistive-thesis-screen-20261015/accessible_qa'
vqa=json.loads((ROOT/'vqa_annotations/val.json').read_text(encoding='utf-8'))
quality={r['image']:r for r in json.loads((ROOT/'quality_annotations/val.json').read_text(encoding='utf-8'))}
bad=[r for r in vqa if not r['answerable']]
rows=[]
for threshold in (2,3):
    n=sum(quality[r['image']]['unrecognizable']>=threshold for r in bad)
    rows.append({'unrecognizable_votes_threshold':threshold,'unanswerable_total':len(bad),
                 'unrecognizable_and_unanswerable':n,'recognizable_and_unanswerable':len(bad)-n,
                 'recognizable_fraction_of_unanswerable':(len(bad)-n)/len(bad)})
result={'purpose':'Threshold sensitivity requested after initial >=3 descriptive result; no model tuning.',
        'limitations':'Recognizable means below this vote threshold; it does not prove missing target, OCR difficulty, or any particular cause of unanswerability.',
        'thresholds':rows}
(ROOT/'quality_threshold_sensitivity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
