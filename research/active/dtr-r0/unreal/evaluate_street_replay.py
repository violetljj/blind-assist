"""Join finished predictions to evaluator-only engine contact samples and render replay.

Metrics are frame-level 3s future contact on this prescribed synthetic walk.
Incomplete future windows are unscored, never counted as true negatives.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,help='Defaults to run/predictions')
    args=parser.parse_args()
    root=args.run.resolve()
    prediction_root=args.predictions or root/'predictions'
    model=json.loads((root/'model/manifest.json').read_text())
    capture=json.loads((root/'capture.json').read_text())
    truth=json.loads((root/'evaluator/frames.json').read_text())
    pred=[json.loads(line) for line in (prediction_root/'frames.jsonl').read_text().splitlines()]
    pred_map={(r['episode_id'],r['sample_index']):r for r in pred}
    truth_map={(r['episode_id'],r['sample_index']):r for r in truth}
    assert len(pred_map)==len(pred)==len(truth_map),'Frame count/identity mismatch'
    horizon=3.0
    results=[]
    panels=[]
    try: font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',18)
    except OSError: font=ImageFont.load_default()
    for episode in model['episodes']:
        eid=episode['episode_id']
        frames=episode['frames']
        ts=[truth_map[(eid,f['sample_index'])] for f in frames]
        ps=[pred_map[(eid,f['sample_index'])] for f in frames]
        assert all(abs(t['time_s']-p['time_s'])<1e-6 for t,p in zip(ts,ps))
        contact_times=[t['time_s'] for t in ts if t['blocking_contact']]
        risk_times=[p['time_s'] for p in ps if p['route_risk']]
        counts={'TP':0,'FP':0,'FN':0,'TN':0,'unscored_tail':0}
        labels=[]
        for t,p in zip(ts,ps):
            if t['time_s']+horizon>ts[-1]['time_s']+1e-6:
                label=None
                counts['unscored_tail']+=1
            else:
                label=any(t['time_s']<=c<=t['time_s']+horizon+1e-6 for c in contact_times)
                counts[('T' if label==p['route_risk'] else 'F')+('P' if p['route_risk'] else 'N')]+=1
            labels.append(label)
        first_contact=min(contact_times) if contact_times else None
        eligible=[t for t in risk_times if first_contact is not None and first_contact-horizon-1e-6<=t<=first_contact]
        result={'episode_id':eid,'frames':len(frames),'sampled_contact_frames':len(contact_times),
                'contact_actors':sorted({t.get('contact_actor') for t in ts if t.get('contact_actor')}),
                'risk_frames':len(risk_times),'first_contact_s':first_contact,
                'first_risk_s':min(risk_times) if risk_times else None,
                'lead_to_first_contact_s':first_contact-min(eligible) if eligible else None,
                'sampled_contacts':contact_times,'frame_counts':counts}
        results.append(result)
        episode_panels=[]
        for f,t,p,label in zip(frames,ts,ps,labels):
            rgb=Image.open(root/'model'/f['rgb_path']).convert('RGB')
            canvas=Image.new('RGB',(960,500),(18,24,31))
            canvas.paste(rgb.resize((640,360)),(0,65))
            depth=np.load(root/'model'/f['depth_path'])
            normalized=np.clip(depth/15,0,1)
            colors=np.stack((255*(1-normalized),180*normalized,255*normalized),axis=-1).astype(np.uint8)
            colors[depth<=0]=0
            canvas.paste(Image.fromarray(colors).resize((320,180)),(640,65))
            d=ImageDraw.Draw(canvas)
            d.text((16,10),f'WILLOW WALK | {eid} | t={f["time_s"]:04.1f}s',font=font,fill='white')
            color='#ff685b' if p['route_risk'] else '#91abbc'
            d.text((16,35),f'X73: {p["event"]} | '+('PREDICTED ROUTE RISK' if p['route_risk'] else 'NO POSITIVE RISK / NOT SAFETY'),font=font,fill=color)
            d.text((652,255),'FORWARD DEPTH 0-15m',font=font,fill='white')
            d.text((652,285),'EVALUATOR ONLY',font=font,fill='#ffc66e')
            d.text((652,310),'Contact now: '+str(t['blocking_contact']),font=font,fill='white')
            d.text((652,335),'Next 3s: '+('UNSCORED' if label is None else str(label)),font=font,fill='white')
            d.text((16,438),'Fixed-time UE RGB-D -> YOLO masks -> retained DTR X73 -> engine contact',font=font,fill='white')
            # Red contact marks and orange model-risk marks on a shared simulation timeline.
            for k,(gt,pr) in enumerate(zip(ts,ps)):
                x=18+int(920*k/max(1,len(ts)-1))
                d.rectangle((x,467,x+14,473),fill='#ff5555' if gt['blocking_contact'] else '#36434c')
                d.rectangle((x,480,x+14,486),fill='#ffbb44' if pr['route_risk'] else '#36434c')
            x=18+int(920*f['sample_index']/max(1,len(ts)-1))
            d.line((x,464,x,490),fill='white',width=2)
            episode_panels.append(canvas)
        chosen=next((i for i,p in enumerate(ps) if p['route_risk']),len(frames)//2)
        episode_panels[chosen].save(root/(eid+'-preview.png'))
        panels.extend(episode_panels)
    summary={'status':'COMPLETE','evidence':'UE synthetic Development integration demonstration',
        'capture_frames':capture['frames'],'horizon_s':horizon,'episodes':results,
        'truth_definition':'UE Visibility-channel capsule blocking overlap sampled at 5Hz, radius0.30m halfheight0.90m; no continuous collision guarantee',
        'tail_rule':'Only frames with full captured 3s future scored',
        'no_risk_rule':'A false model risk flag is not a safety claim; detector coverage is limited',
        'runtime_mode':'Deterministic offline replay; no online feedback or evasive controller',
        'depth_calibration':capture['depth_calibration']}
    (root/'evaluation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # Portable replay, with visible sensor data, prediction and separately labeled truth.
    panels[0].save(root/'replay.gif',save_all=True,append_images=panels[1:],duration=200,loop=0)
    # Self-contained seekable HTML avoids requiring a local web server.
    images=[]
    for panel in panels:
        buf=io.BytesIO()
        panel.save(buf,format='JPEG',quality=82)
        images.append('data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode())
    html='''<!doctype html><meta charset="utf-8"><title>Willow Walk | DTR replay</title>
<style>body{background:#12181f;color:#eef4fa;font:17px system-ui;margin:28px auto;max-width:1000px}img{width:100%}input{width:80%}button{padding:10px 20px}pre{white-space:pre-wrap}</style>
<h1>Willow Walk · 避障链路回放</h1><p>UE 实际 RGB-D → YOLO → DTR X73 → 独立碰撞真值。合成 Development；未接在线控制。</p>
<img id="frame"><p><button id="play">播放 / 暂停</button> <input id="seek" type="range" min="0" value="0"></p>
<details><summary>测量结果与边界</summary><pre id="report"></pre></details>
<script>const images=IMAGES;const summary=SUMMARY;let i=0,playing=false;
const frame=document.getElementById('frame'),seek=document.getElementById('seek');seek.max=images.length-1;
function show(){frame.src=images[i];seek.value=i}seek.oninput=()=>{i=+seek.value;show()};
document.getElementById('play').onclick=()=>playing=!playing;
setInterval(()=>{if(playing){i=(i+1)%images.length;show()}},200);
document.getElementById('report').textContent=JSON.stringify(summary,null,2);show();</script>'''
    (root/'replay.html').write_text(html.replace('IMAGES',json.dumps(images)).replace('SUMMARY',json.dumps(summary)),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
