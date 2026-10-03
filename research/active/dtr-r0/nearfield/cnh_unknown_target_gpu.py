"""Exact signed-CNH likelihood on CUDA; no scene access or scientific scoring.

The CPU reference remains cnh_displacement_ceiling_evaluate.skellam_logpmf_signed.
The positive Skellam series is summed around its largest term, in float64,
until both omitted tails have a geometric bound. No normal approximation or
unbounded truncation is used. Importing this module does not initialize CUDA.
Run --stage check only after the experiment's rendering-complete barrier.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

RELATIVE_TOLERANCE = 1e-13
MAX_TERMS = 100000
DEFAULT_TEMPLATE_BATCH = 4
_KERNEL = None

CUDA_SOURCE = r'''
// NVRTC's CUDA math intrinsics need no host C/C++ headers. The pip runtime may
// have CUDA device headers without a complete Windows compiler include tree.
#define NAN (__longlong_as_double(0x7ff8000000000000ULL))
#define INFINITY (__longlong_as_double(0x7ff0000000000000ULL))

__device__ void fail(int *error, int code, long long index) {
    if (atomicCAS(error, 0, code) == 0) error[1] = (int)index;
}

__device__ void add_positive(double term, double *total, double *compensation) {
    double corrected = term - *compensation;
    double next = *total + corrected;
    *compensation = (next - *total) - corrected;
    *total = next;
}

__device__ double exact_logpmf(int observation, double expected, double background,
                             double tolerance, int max_terms, int *error,
                             long long index) {
    double p = expected + background;
    double n = background;
    double y = (double)observation;
    if (!isfinite(p) || !isfinite(n) || p < 0.0 || n < 0.0) {
        fail(error, 2, index); return NAN;
    }
    if (p == 0.0 && n == 0.0) return y == 0.0 ? 0.0 : -INFINITY;
    if (n == 0.0) {
        if (y < 0.0) return -INFINITY;
        return -p + y*log(p) - lgamma(y + 1.0);
    }
    if (p == 0.0) {
        if (y > 0.0) return -INFINITY;
        return -n - y*log(n) - lgamma(-y + 1.0);
    }
    double order = fabs(y);
    double product = p*n;
    if (!isfinite(product) || product == 0.0) {
        fail(error, 3, index); return NAN;
    }
    // Rationalized positive root avoids cancellation for large signed counts.
    double root = hypot(order, 2.0*sqrt(product));
    double mode = floor((2.0*product)/(root + order));
    if (!isfinite(mode) || mode > 9.0e15) {
        fail(error, 3, index); return NAN;
    }
    int adjustments = 0;
    while (product/((mode+1.0)*(mode+order+1.0)) > 1.0) {
        if (++adjustments > max_terms) { fail(error, 1, index); return NAN; }
        mode += 1.0;
    }
    while (mode > 0.0 && mode*(mode+order)/product > 1.0) {
        if (++adjustments > max_terms) { fail(error, 1, index); return NAN; }
        mode -= 1.0;
    }
    double positive_count = mode + fmax(y, 0.0);
    double negative_count = mode + fmax(-y, 0.0);
    double anchor = -p-n + positive_count*log(p) + negative_count*log(n)
                    -lgamma(positive_count+1.0)-lgamma(negative_count+1.0);
    double total = 1.0, compensation = 0.0;
    double term = 1.0, j = mode;
    bool upper_done = false;
    for (int count=0; count < max_terms; ++count) {
        double ratio = product/((j+1.0)*(j+order+1.0));
        // All subsequent ratios decrease, so this bounds the whole upper tail.
        if (ratio < 1.0 && term*ratio/(1.0-ratio) <= tolerance*0.5*total) {
            upper_done = true; break;
        }
        term *= ratio;
        add_positive(term, &total, &compensation);
        j += 1.0;
    }
    if (!upper_done) { fail(error, 1, index); return NAN; }
    term = 1.0; j = mode;
    bool lower_done = false;
    for (int count=0; count < max_terms; ++count) {
        if (j == 0.0) { lower_done = true; break; }
        double ratio = j*(j+order)/product;
        // Ratios also decrease towards zero downwards from the series mode.
        if (ratio < 1.0 && term*ratio/(1.0-ratio) <= tolerance*0.5*total) {
            lower_done = true; break;
        }
        term *= ratio;
        add_positive(term, &total, &compensation);
        j -= 1.0;
    }
    if (!lower_done) { fail(error, 1, index); return NAN; }
    double answer = anchor + log(total);
    if (!isfinite(answer)) fail(error, 2, index);
    return answer;
}

extern "C" __global__ void skellam_elementwise(
    const int *observed, const double *expected, const double *background,
    double *result, long long size, double tolerance, int max_terms, int *error) {
    long long i = (long long)blockDim.x*blockIdx.x + threadIdx.x;
    if (i < size) result[i] = exact_logpmf(observed[i], expected[i], background[i],
                                         tolerance, max_terms, error, i);
}

extern "C" __global__ void skellam_frame_cells(
    const int *observed, const double *templates, const double *ambient,
    double *result, long long size, int batch_size, double tolerance,
    int max_terms, int *error) {
    long long i = (long long)blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= size) return;
    int cell = (int)(i % 1024);
    int frame = (int)((i / 1024) % 16);
    int candidate = (int)((i / (1024*16)) % batch_size);
    int sample = (int)(i / (1024*16*batch_size));
    int observation_index = sample*16*1024 + frame*1024 + cell;
    int template_index = candidate*16*1024 + frame*1024 + cell;
    double background = 8.0*ambient[frame*64 + cell/16];
    result[i] = exact_logpmf(observed[observation_index], templates[template_index],
                           background, tolerance, max_terms, error, i);
}
'''


def _runtime():
    global _KERNEL
    import cupy as cp
    if _KERNEL is None:
        _KERNEL = cp.RawModule(code=CUDA_SOURCE, options=('--std=c++11',),
                              name_expressions=('skellam_elementwise', 'skellam_frame_cells'))
    return cp, _KERNEL


def _check_parameters(relative_tolerance, max_terms):
    if not np.isfinite(relative_tolerance) or not 0 < relative_tolerance <= RELATIVE_TOLERANCE:
        raise ValueError('Exact likelihood requires relative tolerance in (0, 1e-13]')
    if int(max_terms) != max_terms or not 1 <= max_terms <= np.iinfo(np.int32).max:
        raise ValueError('Positive int32 max_terms required')


def _raise_kernel_error(cp, error):
    code, index = cp.asnumpy(error).tolist()
    if code:
        reason = {1: 'Exact positive series did not meet its geometric tail bound',
                  2: 'Nonfinite exact likelihood/rate',
                  3: 'Poisson rate product/series mode outside float64 domain'}[code]
        raise FloatingPointError(f'{reason}; flattened cell index={index}')


def _observations(values):
    values = np.asarray(values)
    if values.dtype != np.int32:
        if (not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all()
                or np.any(values != np.rint(values))
                or np.any(values < np.iinfo(np.int32).min)
                or np.any(values > np.iinfo(np.int32).max)):
            raise ValueError('Signed int32-range integer observations required')
    return np.ascontiguousarray(values, dtype=np.int32)


def skellam_logpmf_signed_gpu(observed, expected, background, *,
                             relative_tolerance=RELATIVE_TOLERANCE, max_terms=MAX_TERMS):
    """Elementwise CPU-shaped result for accuracy checks; CUDA float64 evaluation.

    Degenerate zero-rate laws retain impossible observations as -inf, matching
    the frozen CPU reference. Numerical convergence failure raises explicitly.
    """
    _check_parameters(relative_tolerance, max_terms)
    y, e, a = np.broadcast_arrays(_observations(observed),
                                  np.asarray(expected, dtype=np.float64),
                                  np.asarray(background, dtype=np.float64))
    if (not np.isfinite(e).all() or not np.isfinite(a).all()
            or np.any(e < 0) or np.any(a < 0)):
        raise ValueError('Finite nonnegative template/background rates required')
    if y.size == 0:
        return np.empty(y.shape, dtype=np.float64)
    cp, module = _runtime()
    arrays = (cp.asarray(np.ascontiguousarray(y)), cp.asarray(np.ascontiguousarray(e)),
              cp.asarray(np.ascontiguousarray(a)))
    output = cp.empty(y.shape, dtype=cp.float64)
    error = cp.zeros(2, dtype=cp.int32)
    module.get_function('skellam_elementwise')(((y.size+255)//256,), (256,),
        (*arrays, output, np.int64(y.size), np.float64(relative_tolerance), np.int32(max_terms), error))
    _raise_kernel_error(cp, error)
    return cp.asnumpy(output)


def frame_likelihood(hist, templates, ambient, *, template_batch=DEFAULT_TEMPLATE_BATCH,
                     relative_tolerance=RELATIVE_TOLERANCE, max_terms=MAX_TERMS):
    """hist[7,4,16,8,8,16], templates[N,16,8,8,16], ambient[16,8,8].

    Returns numpy.float64[28,N,16], summing independent cells for each frame.
    All templates are scored against every sample. The signed-law background
    is 8*ambient, as in the frozen CPU evaluator. No latent count/identity/truth
    arrays are accepted. Batching bounds cell-buffer VRAM (~3.5 MiB/candidate).
    An impossible scene likelihood is reported, never silently omitted.
    """
    _check_parameters(relative_tolerance, max_terms)
    hist, templates, ambient = np.asarray(hist), np.asarray(templates), np.asarray(ambient)
    if (hist.shape != (7, 4, 16, 8, 8, 16) or hist.dtype != np.int32
            or templates.ndim != 5 or templates.shape[1:] != (16, 8, 8, 16)
            or len(templates) == 0 or ambient.shape != (16, 8, 8)):
        raise ValueError('Raw signed int32 CNH/template/ambient axes differ')
    if (not np.isfinite(templates).all() or not np.isfinite(ambient).all()
            or np.any(templates < 0) or np.any(ambient < 0)):
        raise ValueError('Finite nonnegative template/ambient rates required')
    if int(template_batch) != template_batch or template_batch < 1:
        raise ValueError('Positive integer template_batch required')
    cp, module = _runtime()
    d_hist = cp.asarray(np.ascontiguousarray(hist.reshape(28, 16, 1024)))
    d_ambient = cp.asarray(np.ascontiguousarray(ambient, dtype=np.float64))
    result = np.empty((28, len(templates), 16), dtype=np.float64)
    kernel = module.get_function('skellam_frame_cells')
    for begin in range(0, len(templates), int(template_batch)):
        batch = np.ascontiguousarray(templates[begin:begin+int(template_batch)], dtype=np.float64)
        d_templates = cp.asarray(batch)
        cells = cp.empty((28, len(batch), 16, 1024), dtype=cp.float64)
        error = cp.zeros(2, dtype=cp.int32)
        kernel(((cells.size+255)//256,), (256,),
               (d_hist, d_templates, d_ambient, cells, np.int64(cells.size), np.int32(len(batch)),
                np.float64(relative_tolerance), np.int32(max_terms), error))
        _raise_kernel_error(cp, error)
        result[:, begin:begin+len(batch)] = cp.asnumpy(cells.sum(axis=-1, dtype=cp.float64))
        del d_templates, cells, error
    if not np.isfinite(result).all():
        raise ValueError('Unsupported nonfinite scene likelihood; preserve the scene denominator')
    return result


def check(observation_file=None, template_file=None):
    """Synthetic and optional already-rendered scene checks; no score/truth reads."""
    from cnh_displacement_ceiling_evaluate import skellam_logpmf_signed as cpu, TEMPLATE_IDS
    rng = np.random.default_rng(2026100402)
    e = np.exp(rng.uniform(np.log(.001), np.log(10000.), 8192))
    a = np.exp(rng.uniform(np.log(.001), np.log(1000.), 8192))
    y = (rng.poisson(e+a)-rng.poisson(a)).astype(np.int32)
    # Signed remote tails, zero laws, and rates on both sides of a series mode.
    y = np.concatenate((y, np.array([-2000, 2000, -3, 0, 5, 0, 5, -5, 100000], np.int32)))
    e = np.concatenate((e, [0, 0, 0, 0, 0, 5, 5, 5, 2]))
    a = np.concatenate((a, [32, 32, 0, 0, 0, 0, 0, 0, .01]))
    started = time.monotonic()
    expected = cpu(y, e, a)
    actual = skellam_logpmf_signed_gpu(y, e, a)
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], atol=3e-9, rtol=3e-12)
    try:
        skellam_logpmf_signed_gpu(np.int32(0), 0., 1000., max_terms=1)
    except FloatingPointError:
        pass
    else:
        raise AssertionError('Unconverged series accepted')
    # Full shape and candidate-batch alignment with nonconstant frame/zone means.
    ambient = rng.uniform(.02, 3., (16, 8, 8))
    templates = rng.uniform(0., 12., (3, 16, 8, 8, 16))
    p = templates[0]+8*ambient[..., None]
    hist = (rng.poisson(p, size=(7, 4, *p.shape))
            -rng.poisson(8*ambient[..., None], size=(7, 4, *p.shape))).astype(np.int32)
    actual_frame = frame_likelihood(hist, templates, ambient, template_batch=2)
    expected_frame = np.stack([cpu(hist.reshape(28,16,8,8,16), template,
                               8*ambient[...,None]).sum(axis=(-3,-2,-1))
                               for template in templates], axis=1)
    np.testing.assert_allclose(actual_frame, expected_frame, atol=1e-7, rtol=3e-12)
    report = dict(status='PASS_SYNTHETIC', elementwise_n=len(y),
                  elementwise_max_abs_error=float(np.max(np.abs(actual[finite]-expected[finite]))),
                  frame_shape=list(actual_frame.shape),
                  frame_max_abs_error=float(np.max(np.abs(actual_frame-expected_frame))),
                  convergence_failure_checked=True, elapsed_s=time.monotonic()-started)
    cp, _ = _runtime()
    device_id = cp.cuda.runtime.getDevice()
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    name = properties['name']
    report['backend'] = dict(device_id=device_id,
        device_name=name.decode() if isinstance(name, bytes) else str(name),
        compute_capability=[properties['major'], properties['minor']],
        cuda_runtime_version=cp.cuda.runtime.runtimeGetVersion(),
        cuda_driver_version=cp.cuda.runtime.driverGetVersion(), cupy_version=cp.__version__,
        dtype='float64', law='Skellam(expected+8ambient,8ambient)',
        relative_series_tail_tolerance=RELATIVE_TOLERANCE, max_series_terms_per_tail=MAX_TERMS)
    def sha(path):
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    report['module_sha256'] = sha(__file__)
    report['cpu_reference_module_sha256'] = sha(Path(__file__).with_name('cnh_displacement_ceiling_evaluate.py'))
    if bool(observation_file) != bool(template_file):
        raise ValueError('Provide both rendered observation and template files')
    if observation_file:
        with np.load(observation_file, allow_pickle=False) as obs:
            hist, ambient = obs['hist'], obs['ambient']
        with np.load(template_file, allow_pickle=False) as source:
            if 'expected' in source.files:
                templates = source['expected']
                report['rendered_template_format'] = 'expected'
            elif 'endpoints' in source.files:
                # Use the already-frozen prior, not an accuracy-selected subset.
                plan_file = Path(template_file).parent.parent/'PLAN.json'
                prior = json.loads(plan_file.read_text(encoding='utf8'))['prior']
                endpoints = source['endpoints']
                templates = np.stack([endpoints[:,:,0]+rho/.65*(endpoints[:,:,1]-endpoints[:,:,0])
                                      for rho in prior['rhos']], axis=1)
                templates = templates[:,:,TEMPLATE_IDS].reshape(-1,16,8,8,16)
                np.testing.assert_array_equal(source['ambient'], ambient)
                report['rendered_template_format'] = 'frozen prior interpolated endpoints'
                report['rendered_prior_sha256'] = sha(plan_file)
            else:
                raise ValueError('Rendered archive must contain expected or prior-bound endpoints')
        # Deterministic first, middle and last candidates; no truth/score selection.
        indices = sorted(set((0, len(templates)//2, len(templates)-1)))
        selected = templates[indices]
        actual_frame = frame_likelihood(hist, selected, ambient, template_batch=2)
        expected_frame = np.stack([cpu(hist.reshape(28,16,8,8,16), template,
                                   8*ambient[...,None]).sum(axis=(-3,-2,-1))
                                   for template in selected], axis=1)
        np.testing.assert_allclose(actual_frame, expected_frame, atol=1e-7, rtol=3e-12)
        report.update(status='PASS_SYNTHETIC_AND_RENDERED', rendered_candidate_indices=indices,
                      rendered_max_abs_error=float(np.max(np.abs(actual_frame-expected_frame))),
                      rendered_observation=str(Path(observation_file).resolve()),
                      rendered_templates=str(Path(template_file).resolve()),
                      rendered_observation_sha256=sha(observation_file),
                      rendered_templates_sha256=sha(template_file))
    report['elapsed_s'] = time.monotonic()-started
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check',))
    parser.add_argument('--observation-file')
    parser.add_argument('--template-file')
    parser.add_argument('--report-file')
    args = parser.parse_args()
    report = check(args.observation_file, args.template_file)
    if args.report_file:
        # A verification record is evidence; never overwrite a previous check.
        with Path(args.report_file).open('x', encoding='utf8') as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
