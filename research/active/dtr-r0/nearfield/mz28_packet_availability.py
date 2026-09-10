"""Fixed TRAIN packet-coordinate knownness retrieval; not occupancy or CLEAR."""
import torch
from torch import nn


def packet4(ranges,valid):
    good=valid.bool()&torch.isfinite(ranges)&(ranges>0)&(ranges<=4)
    return torch.cat([torch.where(good,ranges,torch.zeros_like(ranges))/4,good.float()],-1).float()


class PacketAvailability(nn.Module):
    """Current inputs are sensor packet only; all labels belong to fixed TRAIN."""
    def __init__(self,ids,packets,known):
        super().__init__()
        assert ids.ndim==1 and len(ids)>=5 and (ids[1:]>ids[:-1]).all()
        assert packets.shape==(len(ids),64,4) and known.shape==(len(ids),64,49)
        self.register_buffer('ids',ids.long())
        self.register_buffer('packets',packets.permute(1,0,2).double().contiguous())
        self.register_buffer('known',known.permute(1,0,2).to(torch.uint8).contiguous())

    @torch.no_grad()
    def forward(self,ranges,valid):
        p=packet4(ranges,valid).double();distance=torch.zeros((*p.shape[:2],len(self.ids)),device=p.device,dtype=torch.float64)
        for c in range(4):distance.add_((p[:,:,c,None]-self.packets[None,:,:,c]).square())
        index=[];values=[]
        for _ in range(5):
            # min returns the first bank position on ties; bank globalIDs ascend.
            value,idx=distance.min(-1);index.append(idx);values.append(value)
            distance.scatter_(-1,idx[...,None],float('inf'))
        index=torch.stack(index,-1);values=torch.stack(values,-1).sqrt()
        zone=torch.arange(64,device=p.device)[None,:,None].expand_as(index)
        votes=self.known[zone,index].sum(2).to(torch.uint8)
        return dict(availability=votes.float()-2.5,votes=votes,neighbor_ids=self.ids[index],neighbor_distances=values)
