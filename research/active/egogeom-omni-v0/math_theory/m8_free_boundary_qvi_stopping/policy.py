"""Finite encounter optimal stopping with a binary heteroscedastic observation."""
import numpy as np
from scipy.special import expit, ndtr

DIST=.5+.25*np.arange(39)
PROB=np.linspace(0.,1.,1001)
PRICE=.18
LOSS=10.
SIGMA=.06
DELAY_STEPS=3


def hazards():
    # Conditional collision hazard per 0.2 s if obstacle exists and no intervention.
    out=.42*expit((2.-DIST)/.36)
    out[0]=1.
    return out


def positive_likelihood(coef=SIGMA):
    # Midpoint test with N(+/-0.5, (coef*d^2)^2); variance scales as d^4.
    return ndtr(.5/(coef*DIST**2))


def delay_risk(h):
    return np.array([1.-np.prod(1.-h[max(0,i-DELAY_STEPS+1):i+1]) for i in range(len(h))])


def observe(prior, positive, likelihood):
    l1=np.where(positive,likelihood,1.-likelihood)
    l0=1.-l1
    return prior*l1/np.maximum(prior*l1+(1.-prior)*l0,1e-15)


def solve(price=PRICE, grid=PROB, coef=SIGMA):
    h=hazards()
    lk=positive_likelihood(coef)
    insured=delay_risk(h)
    value=np.zeros((len(DIST),len(grid)))
    advantage=np.zeros_like(value)
    cut=[]
    for i,haz in enumerate(h):
        immediate=price+LOSS*insured[i]*grid
        wait=LOSS*haz*grid
        if i:
            survival=1.-grid*haz
            prior=grid*(1.-haz)/survival
            pos=prior*lk[i-1]+(1-prior)*(1-lk[i-1])
            for y,py in ((True,pos),(False,1.-pos)):
                post=observe(prior,y,lk[i-1])
                wait+=survival*py*np.interp(post,grid,value[i-1])
        advantage[i]=wait-immediate
        value[i]=np.minimum(immediate,wait)
        mask=advantage[i]>=0.
        # In a binary experiment, stop cost affine and continuation cost concave:
        # the stop set is an interval, possibly empty.
        indices=np.flatnonzero(mask)
        if len(indices):
            assert np.all(mask[indices[0]:indices[-1]+1])
        cut.append([float(grid[indices[0]]),float(grid[indices[-1]])] if len(indices) else None)
    return value,advantage,cut


class QVIAlertPolicy:
    """A host-side diagnostic. Map historical BEV inputs to model beliefs with no labels."""
    def __init__(self,advantage):
        self.advantage=advantage
        self.filtered=0.
        self.last=0.

    def step(self,raw,bev_dist,single_dist,sigma):
        # BEV risk has arbitrary units; this bounded map is a proxy, not a calibrated posterior.
        current=float(np.clip(raw/(raw+.6),0.,1.))
        self.filtered=.65*self.filtered+.35*current
        distance=float(np.clip(bev_dist if np.isfinite(bev_dist) else single_dist,DIST[0],DIST[-1]))
        # Decrease reliance on an uncertain single-frame reading.
        weight=1./(1.+sigma/(distance+.1))
        p=weight*current+(1.-weight)*self.filtered
        i=int(np.clip(np.rint((distance-.5)/.25),0,len(DIST)-1))
        return float(np.interp(p,PROB,self.advantage[i]))
