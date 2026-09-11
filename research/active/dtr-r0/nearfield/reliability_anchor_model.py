"""Three observable neighbor-geometry proxies; not sensor quality or confidence.

No signal/sigma/status/label is inferred. Fixed four-connected 8x8 adjacency.
Both arms have the same expanded head; BASE's extra inputs are always zero.
"""
import torch
from torch import nn
from torch.nn import functional as F
from mz56_global_anchor_model import AnchorQuery,loss_for


class ReliabilityAnchorQuery(AnchorQuery):
    def __init__(self,initial_mz54_state,initial_mz70_state,arm):
        if arm not in ('BASE','RELIABILITY'):raise ValueError(arm)
        super().__init__(initial_mz54_state,'GLOBAL_ANCHOR')
        self.load_state_dict(initial_mz70_state,strict=True)
        old=self.head[0]
        expanded=nn.Linear(52,32)
        with torch.no_grad():
            expanded.weight[:,:49].copy_(old.weight)
            expanded.weight[:,49:].zero_();expanded.bias.copy_(old.bias)
        self.head[0]=expanded
        self.augmentation_arm=arm
        indices=[]
        for row in range(8):
            for col in range(8):
                indices.append([r*8+c if 0<=r<8 and 0<=c<8 else 64
                                for r,c in [(row-1,col),(row+1,col),(row,col-1),(row,col+1)]])
        indices=torch.tensor(indices,dtype=torch.long)
        self.register_buffer('neighbor_indices',indices)
        self.register_buffer('neighbor_degree',(indices<64).sum(1))

    def neighbor_geometry(self,ranges,valid):
        """Return [B,64,3] and availability [B,64], from observations alone.

        Channel order: (nearest valid range - neighbor lower median)/4,
        neighbor lower-median absolute deviation/4, valid-neighbor fraction.
        Use each neighbor's nearest valid distance. Missing/out-of-grid neighbors
        never enter a statistic. All channels zero when center or all neighbors
        are missing; an isolated valid zone does not imply reliable geometry.
        This is scene geometry, not uncertainty or a reason to discard an echo.
        """
        if ranges.ndim!=3 or ranges.shape[1:]!=(64,2) or valid.shape!=ranges.shape:
            raise ValueError('Expected ranges/valid [B,64,2]')
        good=valid.bool()&torch.isfinite(ranges)&(ranges>0)&(ranges<=4)
        nearest=ranges.masked_fill(~good,torch.inf).amin(-1)
        center_good=good.any(-1)
        padded=torch.cat((nearest,nearest.new_full((len(ranges),1),torch.inf)),1)
        neighbors=padded[:,self.neighbor_indices]
        neighbor_good=torch.isfinite(neighbors)
        observed=neighbors.masked_fill(~neighbor_good,torch.nan)
        median=observed.nanmedian(-1).values
        mad=(observed-median[...,None]).abs().nanmedian(-1).values
        available=center_good&neighbor_good.any(-1)
        residual=torch.where(available,(nearest-median)/4,torch.zeros_like(nearest))
        dispersion=torch.where(available,mad/4,torch.zeros_like(nearest))
        fraction=neighbor_good.sum(-1).to(ranges.dtype)/self.neighbor_degree[None]
        fraction=torch.where(available,fraction,torch.zeros_like(fraction))
        proxies=torch.stack((residual,dispersion,fraction),-1)
        return proxies,available

    def inspect(self,normalized_features,ranges,valid,suppress_global=False):
        if normalized_features.ndim!=4 or tuple(normalized_features.shape[1:])!=(64,45,80):
            raise ValueError('Expected FULL features [B,64,45,80]')
        batch=len(normalized_features)
        if len(ranges)!=batch:raise ValueError('Feature and packet batch sizes differ')
        # Exact original AnchorQuery concatenation through its first49 features.
        local=self.local(normalized_features)
        grid=self.grid[None].expand(batch,-1,-1,-1)
        tokens=F.grid_sample(local,grid,align_corners=False).permute(0,2,3,1)
        tokens=tokens.masked_fill(~self.candidate_mask[None,:,:,None],0.)
        sums=tokens.new_zeros((batch,len(self.context_counts),16))
        ids=self.context_ids[None,:,None].expand(batch,-1,16)
        sums.scatter_add_(1,ids,tokens.reshape(batch,-1,16))
        means=sums/self.context_counts.clamp_min(1)[None,:,None]
        context=means[:,self.context_ids].reshape(batch,45,80,16)
        packet,available=self.packet_features(ranges,valid)
        static=torch.cat((self.absolute,self.relative),-1)[None].expand(batch,-1,-1,-1)
        coverage=self.sensor_coverage[None,:,:,None].expand(batch,-1,-1,-1)
        observed,anchor_available=self.encode_anchor(ranges,valid)
        anchor=observed if not suppress_global else torch.zeros_like(observed)
        expanded=anchor[:,None,None].expand(-1,45,80,-1)
        geometry,geometry_available=self.neighbor_geometry(ranges,valid)
        supplied=geometry if self.augmentation_arm=='RELIABILITY' else torch.zeros_like(geometry)
        supplied=torch.cat((supplied,supplied.new_zeros((batch,1,3))),1)
        zone=torch.where(self.sensor_coverage,self.sensor_zone,64)
        raster_geometry=supplied[:,zone]
        field=self.head(torch.cat((tokens,context,static,packet,coverage.to(tokens.dtype),expanded,raster_geometry),-1))
        field=field.masked_fill(~self.candidate_mask[None,:,:,None],0.)
        return dict(field=field,candidate_mask=self.candidate_mask,OPEN=self.pool(field,self.candidate_mask),
            CROP=self.pool(field,self.candidate_mask&self.crop_mask),sensor_coverage=self.sensor_coverage,
            packet_available=available,observed_anchor=observed,anchor_vector=anchor,anchor_available=anchor_available,
            geometry_proxies=geometry,geometry_available=geometry_available,
            interpretation='PREDICTED_LOCAL_INTRUSION_WITH_OBSERVABLE_GEOMETRY_NOT_HARDWARE_QUALITY')
