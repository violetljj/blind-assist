"""Summarize a completed field batch and render every frame for scene review."""
import argparse
from collections import Counter
import json
from pathlib import Path


def build(collection):
    from PIL import Image, ImageDraw
    collection=Path(collection).resolve(strict=True)
    output=collection/'report'; output.mkdir(exist_ok=True)
    validation=json.loads((collection/'collection-validation.json').read_text())
    routes=[]
    for path in sorted(collection.glob('*/*-bundle.json')):
        bundle=json.loads(path.read_text()); frames=bundle['frames']
        rid=bundle['route_id']; cols=4; width=320; height=204
        sheet=Image.new('RGB',(cols*width,((len(frames)+cols-1)//cols)*height),'#181818')
        draw=ImageDraw.Draw(sheet)
        for n,frame in enumerate(frames):
            x=n%cols*width; y=n//cols*height
            with Image.open(frame['rgb']['path']) as source:
                sheet.paste(source.convert('RGB').resize((320,180)),(x,y+24))
            draw.text((x+4,y+5),f"{frame['sample_index']:02d} {frame['variant']} {frame['labels'].get('near')}",fill='white')
        sheet.save(output/(rid+'.jpg'),quality=92)
        active=[t for f in frames for t in f['targets'] if t['label_status']!='NOT_PRESENT']
        routes.append(dict(route_id=rid,split=bundle['split'],scene_type=bundle['scene_type'],
            frames=len(frames),paired=sum(f['synchronization']['status']=='PAIRED' for f in frames),
            ready=sum(f['source_load_status']=='READY' for f in frames),
            near_labels=dict(Counter(str(f['labels'].get('near')) for f in frames)),
            active_target_status=dict(Counter(t['label_status'] for t in active)),
            uncertain_targets=[dict(sample_index=f['sample_index'],target_id=t['target_id'])
                for f in frames for t in f['targets'] if t['label_status']=='UNKNOWN'],
            contact_sheet=str(output/(rid+'.jpg'))))
    result=dict(status=validation['status'],frames=sum(r['frames'] for r in routes),routes=routes,
        issue_counts=dict(Counter(i['code'] for i in validation['issues'])),
        scope='Static settled City Sample acquisition; controlled supported fixtures; no model evaluation')
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plan=json.loads((collection/'plan.json').read_text())
    fig,ax=plt.subplots(figsize=(12,6))
    colours=dict(train='#2374ab',dev='#e07a25',test='#2f9564')
    for region in plan['regions']:
        x0,y0,x1,y1=region['bounds_xy_m']; colour=colours[region['split']]
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor=colour,edgecolor=colour,alpha=.10))
        ax.text((x0+x1)/2,y0+5,region['region_id']+' / '+region['split'],ha='center',color=colour)
    for route in plan['routes']:
        points=[p['camera'] for p in route['waypoints']]
        xs=[p['x'] for p in points];ys=[p['y'] for p in points]
        ax.plot(xs,ys,'.-',color=colours[route['split']])
        offsets=dict(sidewalk=18,intersection=-15,narrow_passage=35)
        ax.annotate(route['scene_type'].replace('_',' '),(xs[len(xs)//2],ys[len(ys)//2]),
            xytext=(0,offsets[route['scene_type']]),textcoords='offset points',ha='center',fontsize=8,
            arrowprops=dict(arrowstyle='-',color='#777777',lw=.6))
    ax.set(xlabel='UE world X (m)',ylabel='UE world Y (m)',title='City field v1: region-separated routes (shared city assets)')
    ax.autoscale_view();ax.set_aspect('equal');ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(output/'route-map.png',dpi=150);plt.close(fig)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection',type=Path)
    build(parser.parse_args().collection)
