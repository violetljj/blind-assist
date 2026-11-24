"""Hutchinson log-depth sensitivity and a foldable LayerNorm-to-MatMul diagonal gauge."""
import gc
import json
import re
import sys

import numpy as np
import onnx
from onnx import numpy_helper

from eval_empirical import ASSETS, OUT, ROUTE, convert, build, dump, model_path, sha


def fisher():
    import torch
    sys.path[:0]=[str(ROUTE),str(ROUTE.parent/"egoclear-v0")]
    from egogeom_omni_v41_model import EgoGeomOmniV41Net,EgoGeomV41Config
    torch.set_num_threads(2)
    torch.manual_seed(13024)
    ck_path=ASSETS/"egogeom_omni_v41_pure_rgb_best.pt"
    ck=torch.load(ck_path,mmap=True,map_location="cpu",weights_only=False)
    model=EgoGeomOmniV41Net(EgoGeomV41Config(**ck["cfg"]),pretrained_backbone=False)
    model.tof_branch=None
    model.load_state_dict({k:v for k,v in ck["ema"].items() if not k.startswith("tof_branch.")},strict=True)
    del ck
    model.eval().requires_grad_(False)
    stats={}
    handles=[]
    channels={}
    used=json.loads((OUT/"calibration.json").read_text())
    def observe(name,module):
        def hook(mod,args,h):
            if not isinstance(h,torch.Tensor) or not h.requires_grad:
                return
            item=stats.setdefault(name,{"min":float("inf"),"max":float("-inf"),"trace":0.,"forward":0,"backward":0})
            t=h.detach()
            item["min"]=min(item["min"],t.min().item())
            item["max"]=max(item["max"],t.max().item())
            item["forward"]+=1
            local=name.startswith("enc.") and (name.endswith(".norm") or name.endswith(".mlp.fc1"))
            if local:
                x=t.reshape(-1,t.shape[-1])
                c=channels.setdefault(name,{"second":np.zeros(x.shape[-1]),"grad":np.zeros(x.shape[-1])})
                c["second"]+=x.square().mean(0).numpy()
            def record(g):
                squared=g.detach().square()
                item["trace"]+=squared.sum().item()
                item["backward"]+=1
                if local:
                    c["grad"]+=squared.reshape(-1,squared.shape[-1]).sum(0).numpy()
            h.register_hook(record)
        return hook
    for name,mod in model.named_modules():
        if name and not list(mod.children()) and not isinstance(mod,torch.nn.Identity):
            handles.append(mod.register_forward_hook(observe(name,mod)))
    for r in used:
        d=OUT/"cal"/r["clip"]
        def tensor(k,shape):
            return torch.from_numpy(np.fromfile(d/(k+".raw"),"<f4").reshape(shape))
        x=tensor("rgb",(1,3,288,384)).requires_grad_()
        rays=tensor("grav_rays",(1,3,288,384))
        focal=tensor("host_focal",(1,1,1,1))
        z=model(x,rays,None,None,host_focal=focal)["log_depth"]
        v=torch.empty_like(z).bernoulli_(.5).mul_(2).sub_(1)
        (z*v).sum().div(z.numel()**.5).backward()
    for handle in handles:
        handle.remove()
    scales={}
    for name,c in channels.items():
        if not name.endswith(".norm"):
            continue
        fc=name[:-5]+".mlp.fc1"
        layer=model.get_submodule(fc)
        dw=(2*layer.weight.detach().abs().max().item()/255)
        dh=(stats[name]["max"]-stats[name]["min"])/255
        a=c["grad"]*dh**2
        b=c["second"]/len(used)*channels[fc]["grad"].sum()*dw**2
        ratio=np.divide(a,b,out=np.ones_like(a),where=b>0)
        s=np.clip(np.round(.25*np.log2(np.maximum(ratio,2**-32))),-3,3)
        scales[name]=np.exp2(s).tolist()
    for m in stats.values():
        m["trace"]/=len(used)
        m["estimated_uniform_a8_loss"]=m["trace"]*((m["max"]-m["min"])/255)**2/12
    assert len(scales)==36,len(scales)
    dump(OUT/"fisher.json",{"checkpoint_sha256":sha(ck_path),"frames":len(used),"probes_per_frame":1,
                           "leaf_modules":stats,"diagonal_scales":scales})
    return scales


def export(kind):
    scales=(json.loads((OUT/"fisher.json").read_text())["diagonal_scales"] if (OUT/"fisher.json").exists() else fisher())
    g=onnx.load(model_path(kind)).graph
    users={}
    for n in g.node:
        for x in n.input:
            users.setdefault(x,[]).append(n)
    init={t.name:t for t in g.initializer}
    count=0
    for norm in g.node:
        if norm.op_type!="LayerNormalization" or len(norm.input)!=3:
            continue
        name=norm.input[1].removeprefix("core.").removesuffix(".weight")
        if name not in scales:
            continue
        following=users[norm.output[0]]
        assert len(following)==1 and following[0].op_type=="MatMul"
        names=[norm.input[1],norm.input[2],following[0].input[1]]
        s=np.asarray(scales[name],dtype="float32")
        for i,k in enumerate(names):
            assert len(users[k])==1
            w=numpy_helper.to_array(init[k])
            adjusted=(w*s if i<2 else w/s[:,None])
            init[k].CopyFrom(numpy_helper.from_array(adjusted,k))
        count+=1
    assert count==36
    m=onnx.load(model_path(kind))
    m.graph.CopyFrom(g)
    del g
    onnx.checker.check_model(m)
    out=OUT/f"equal_{kind}.onnx"
    onnx.save(m,out)
    del m
    gc.collect()
    import onnxruntime as ort
    opts=ort.SessionOptions()
    opts.intra_op_num_threads=1
    predictions=[]
    for path in [model_path(kind),out]:
        sess=ort.InferenceSession(str(path),opts,providers=["CPUExecutionProvider"])
        results=[]
        for rec in json.loads((OUT/"calibration.json").read_text()):
            d=OUT/"cal"/rec["clip"]
            f={i.name:np.fromfile(d/(i.name+".raw"),"<f4").reshape(i.shape) for i in sess.get_inputs()}
            results.append(sess.run(None,f))
        predictions.append(results)
        del sess
    maximum={j:max(float(np.max(np.abs(a[j]-b[j]))) for a,b in zip(*predictions)) for j in range(4)}
    dump(OUT/f"equal_{kind}.json",{"blocks":count,"onnx_sha256":sha(out),"max_host_fp32_abs_4_outputs":maximum})
    assert max(maximum.values()) < 3e-5,maximum


def mixed(kind):
    """Allocate one proxy unit per selected eligible MLP site; use only calibration encodings."""
    data=json.loads((OUT/"fisher.json").read_text())
    base=json.loads((OUT/f"native_{kind}_a8"/"graph_encoding.json").read_text())
    # fc2 output includes bias/residual etc.; only request encodings present in the actual converter dump.
    candidates=[]
    for name,metric in data["leaf_modules"].items():
        if not name.startswith("enc.") or not name.endswith(("mlp.fc1","mlp.fc2")):
            continue
        # Path names in exported timm layer-list modules may include repeated "blocks".
        stage,block=re.search(r"stages_(\d+)\.blocks\.(\d+)\.",name).groups()
        prefix=f"/core/enc/stages_{stage}/blocks/blocks.{block}/mlp/"+name.rsplit(".",1)[-1]+"/"
        matches=[k for k in base["activation_encodings"] if k==prefix+"Add_output_0"]
        if len(matches)==1:
            candidates.append((metric["estimated_uniform_a8_loss"],matches[0]))
    assert len(candidates)>=16,len(candidates)
    selected=sorted(candidates,reverse=True)[:16]
    enc={}
    for _,target in selected:
        orig=base["activation_encodings"][target][0]
        scale=(orig["max"]-orig["min"])/65535
        offset=round(orig["min"]/scale)
        enc[target]=[dict(orig,bitwidth=16,scale=scale,offset=offset,min=offset*scale,max=(65535+offset)*scale)]
    # These MLP output tensors are downstream of exactly compensated diagonal folds.
    folder=OUT/("mixed_"+kind)
    folder.mkdir(exist_ok=True)
    path=folder/"overrides.json"
    dump(path,{"activation_encodings":enc,"param_encodings":{}})
    dump(folder/"allocation.json",{"eligible":len(candidates),"proxy_budget_units":16,"selected":selected})
    convert(kind,"mixed")
    build(kind,8,"mixed")
