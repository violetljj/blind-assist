"""v4 typed evidence contract; no parameter fitting, learned weights or cuts."""
from __future__ import annotations
import numpy as np


def real_adapter(hist, ambient, signed_bias, noise):
    h=np.asarray(hist,np.float64); b=np.asarray(signed_bias,np.float64)
    a=np.asarray(ambient,np.float64)
    if not all(np.isfinite(v).all() for v in (h,a,b)):
        raise ValueError('UNKNOWN: nonfinite decoded input/background')
    physical=np.maximum(b,0)
    residual=h-np.minimum(physical,np.maximum(h,0))
    ca,cb,cc=noise['adapter_coefficients_a_b_c']
    variance=np.maximum(ca*physical+cb*np.maximum(a,0)[...,None]+cc,
                        noise['numerical_variance_floor'])
    return (residual/np.sqrt(variance)).astype(np.float32)


def sim_adapter(hist, ambient, bias):
    """Signed ambient-difference input, exact inherited FP32->FP16 arithmetic."""
    h=np.asarray(hist,np.float32);b=np.asarray(bias,np.float32)
    a=np.asarray(ambient,np.float32)
    if np.any(b<0) or not all(np.isfinite(v).all() for v in (h,a,b)):
        raise ValueError('UNKNOWN: simulation needs finite nonnegative physical bias')
    z=((h-b)/np.sqrt(np.maximum(16*a[...,None]+np.maximum(b,0),1e-9))).astype(np.float16)
    blank=np.all(h==0,axis=(-3,-2,-1))|np.all(h==b,axis=(-3,-2,-1))
    z[blank]=0
    return z


def native_evidence(z,drop_bins=()):
    """Only the inherited mask; exact absence is a representation, not a cut."""
    z=np.array(z,copy=True)
    if z.shape[-3:]!=(8,8,16) or not np.isfinite(z).all():
        raise ValueError('UNKNOWN: finite native8x8x16 input required')
    if tuple(drop_bins) not in ((),(0,),(0,1)):
        raise ValueError('Only previously sealed candidate masks allowed')
    if drop_bins:z[...,tuple(drop_bins)]=0
    positive=np.any(z>0,axis=(-3,-2,-1))
    z[~positive]=0
    return z,positive


def history_support(frame_positive,lengths):
    positive=np.asarray(frame_positive,bool);lengths=np.asarray(lengths,int)
    if positive.ndim!=1 or lengths.shape!=positive.shape:
        raise ValueError('Need matching causal frame/history vectors')
    starts=np.arange(len(positive))+1-lengths
    if np.any(starts<0) or np.any(lengths<1) or np.any(lengths>8):
        raise ValueError('Invalid inherited history/reset lengths')
    prefix=np.r_[0,np.cumsum(positive.astype(int))]
    return prefix[1:]-prefix[starts]>0


class ContractReplay:
    """Original frozen inference, plus explicit unavailable evidence scores."""
    def __init__(self,manifest=None,**kwargs):
        import cnh_sim2real_frozen as F
        from cnh_no_bin0_dr_replay import DesignReplay
        self.engine=F.FrozenReplay(**kwargs) if manifest is None else DesignReplay(manifest,**kwargs)
        self.drop_bins=() if manifest is None else self.engine.drop_bins

    def infer(self,z,seq,ms,check=lambda:None):
        native,positive=native_evidence(z,self.drop_bins)
        result=self.engine.infer(native,seq,ms,check)
        present=history_support(positive,result['history'])
        for key in ('m3','old5','S_grade','Sensemble'):
            result['ungated_'+key]=result[key].copy()
            result[key]=np.where(present[:,None],result[key],0).astype(result[key].dtype)
        result['frame_positive']=positive
        result['evidence_present']=present
        result['score_evidence_valid']=present[:,None]&result['eligible'][:,None]
        return result

    def close(self):self.engine.close()
