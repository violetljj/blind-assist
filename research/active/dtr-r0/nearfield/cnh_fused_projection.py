"""Fused CUDA (CuPy) version of the CVR history projection used by M3 features.

Same algebra as cnh_dual_sensor_envelope_natural.Predictor.paired_project in FP32:
one thread per (sequence, voxel) loops over history frames and the 27 sub-points,
so the [C*L, 363k, 3] intermediates are never materialised. Summation order
differs from the torch path; parity is checked by `check()` (voxels and M3 states).
"""
import numpy as np

KERNEL = r'''
extern "C" __global__ void project(const float* pts, const int nvox, const float* mats,
    const float* z, const float* vol, const float vv, const float edge, const float width,
    const int C, const int L, float* out)
{
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid >= C*nvox) return;
    int c = gid / nvox, v = gid - c*nvox;
    float tot = 0.f, cnt = 0.f, lst = 0.f;
    for (int l = 0; l < L; ++l) {
        const float* t = mats + (c*L + l)*16;
        const float* zz = z + (c*L + l)*1024;
        float ev = 0.f; int nvalid = 0;
        for (int s = 0; s < 27; ++s) {
            const float* p = pts + (v*27 + s)*3;
            float dx = p[0]-t[3], dy = p[1]-t[7], dz = p[2]-t[11];
            float qx = dx*t[0] + dy*t[4] + dz*t[8];
            float qy = dx*t[1] + dy*t[5] + dz*t[9];
            float qz = dx*t[2] + dy*t[6] + dz*t[10];
            float r = sqrtf(qx*qx + qy*qy + qz*qz);
            float zc = fmaxf(qz, 1e-30f);
            float f0 = floorf((qx/zc + edge)/(2.f*edge)*8.f);
            float f1 = floorf((qy/zc + edge)/(2.f*edge)*8.f);
            float fb = floorf(r/width);
            if (qz > 0.f && f0 >= 0.f && f0 < 8.f && f1 >= 0.f && f1 < 8.f && fb >= 0.f && fb < 16.f) {
                int idx = ((int)f1*8 + (int)f0)*16 + (int)fb;
                ev += zz[idx]*(vv/27.f/vol[idx]);
                ++nvalid;
            }
        }
        tot += ev; cnt += nvalid/27.f;
        if (l == L-1) lst = ev;
    }
    out[(c*3 + 0)*nvox + v] = tot;
    out[(c*3 + 1)*nvox + v] = cnt;
    out[(c*3 + 2)*nvox + v] = lst;
}
'''


class FusedProjector:
    def __init__(self, projector):
        import cupy as cp
        from cnh_cvr_projection import SUB, SHAPE, EDGE, WIDTH
        assert SUB == 3
        self.cp = cp; self.shape = SHAPE
        self.pts = cp.asarray(projector.points.detach().cpu().numpy().astype(np.float32))
        self.vol = cp.asarray(projector.volumes.detach().cpu().numpy().astype(np.float32))
        self.nvox = int(np.prod(SHAPE)); assert self.pts.shape[0] == self.nvox*27
        self.vv, self.edge, self.width = np.float32(projector.voxel_volume), np.float32(EDGE), np.float32(WIDTH)
        self.kernel = cp.RawKernel(KERNEL, 'project', options=('--std=c++11', '--fmad=false'))

    def __call__(self, z, matrices):
        """z[C,L,8,8,16] (any float), matrices[C,L,4,4] -> torch float32 [C,3,24,17,33] on cuda."""
        import torch
        cp = self.cp; C, L = matrices.shape[:2]
        m = cp.asarray(np.ascontiguousarray(matrices, dtype=np.float32).reshape(C, L, 16))
        zz = cp.asarray(np.ascontiguousarray(np.asarray(z, dtype=np.float32).reshape(C, L, 1024)))
        out = cp.empty((C, 3, self.nvox), dtype=cp.float32)
        n = C*self.nvox
        self.kernel(((n+255)//256,), (256,), (self.pts, np.int32(self.nvox), m, zz, self.vol, self.vv,
                    self.edge, self.width, np.int32(C), np.int32(L), out))
        return torch.as_tensor(out, device='cuda').reshape(C, 3, *self.shape)
