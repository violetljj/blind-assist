"""Fixed eighteen-view functional geometry source, independent of outcomes."""
import argparse
import json
from pathlib import Path


def cases():
    camera=dict(x=26.,y=0.,z=1.82,pitch=-10.,yaw=0.,roll=0.)
    material="/Game/StreetLab/Materials/Bronze"
    def box(name,x,y,z,sx,sy,sz,kind="cube",mat=material):
        return dict(name=name,kind=kind,center_m=[x,y,z],size_m=[sx,sy,sz],material=mat)
    def shape(family,distance):
        x=26+distance
        if family=="low": return [box("low",x,0,.22,.4,.6,.2)]
        if family=="pole": return [box("pole",x,0,1.02,.04,.04,1.8,kind="cylinder")]
        if family=="bar": return [box("bar",x,0,1.72,.08,1.2,.08)]
        if family=="wall": return [box("wall",x,0,1.02,.12,1.5,1.8)]
        if family=="irregular": return [box("stem",x,-.18,.92,.18,.12,1.6),box("foot",x,.06,.27,.18,.6,.3)]
        raise ValueError(family)
    rows=[]
    def add(name,objects,**meta):
        pose={**camera,**meta.pop("camera",{})}
        rows.append(dict(name=name,camera=pose,objects=objects,**meta))
    add("ordinary_ground",[],family="ground",condition="control")
    for family in ("low","pole","bar","wall","irregular"):
        for condition,distance in (("near",2.),("far",5.)):
            add(f"{family}_{condition}",shape(family,distance),family=family,
                condition=condition,pair=family+"_distance")
    for pitch in (-25.,25.):
        add("low_pitch_"+str(int(pitch)),shape("low",2.),family="low",condition="pitch",
            camera=dict(pitch=pitch),pair="low_pitch")
    for distance in (8.,18.):
        add("low_background_"+str(int(distance)),shape("low",2.)+
            [box("background",26+distance,0,2.12,.15,4.,4.,mat="/Game/StreetLab/Materials/Limestone")],
            family="low",condition="background",pair="low_background")
    add("wall_ground_hidden",[box("close_wall",27.,0,2.12,.12,6.,4.)],
        family="wall",condition="ground_hidden")
    add("flat_marking",[box("flush_mark",28.,0,.121,.6,.6,.002,
                            mat="/Game/StreetLab/Materials/Charcoal")],family="ground",condition="marking")
    add("low_transition",shape("low",3.),family="low",condition="transition")
    assert len(rows)==18 and len({r["name"] for r in rows})==18
    return rows


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[4]
    output=args.output.resolve()
    if output.exists() or not output.is_relative_to((root/"artifacts.local").resolve()):
        raise ValueError("Require new canonical artifact file")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(schema="nearfield-distinct-view-spec-v1",cases=cases()),indent=2),encoding="utf-8")
