"""Fixed MZ15 QUERY readout, with sampling calibrated to the common RGB ROI."""
import torch
from mz15_shared_support import LocalSupportReadout


def make_model(initial):
    model=LocalSupportReadout(shared=False)
    model.load_state_dict(initial)
    # Crop208:432,68:292 has the same optical center as native640x360.
    # Physical rays remain unchanged. Grid and absolute input coordinates refer
    # to the same224x224 crop in both resolution arms.
    with torch.no_grad():
        model.grid.mul_(model.grid.new_tensor([640/224,360/224]))
    return model
