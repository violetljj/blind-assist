"""Run the symbolic checks, discrete quantizers and isolated-head certificates."""

import json

import numpy as np
import sympy as sp
from quant_head import HALF_SPAN, HI, LO, OUT, decode, logcode, mixed_primitive
from scipy.integrate import quad


def check_symbolic():
    z, a, b, v, p = sp.symbols("z a b v p", positive=True)
    k = sp.sqrt(b / a)
    primitive = sp.sqrt(a) * (sp.asinh(z / k) - sp.sqrt(z**2 + k**2) / z)
    risk = a / z**2 + b / z**4
    assert sp.simplify(sp.diff(primitive, z)**2 - risk) == 0
    # Euler-Lagrange for F=integral rho du, L=pw/(12 F'^2).
    assert sp.simplify(sp.diff(p / (12 * v**2), v, 2) - p / (2 * v**4)) == 0
    pade = (12 + 6*z + z**2) / (12 - 6*z + z**2)
    assert sp.simplify(sp.diff(pade, z) / pade - 1 + z**4/(144 - 12*z**2 + z**4)) == 0


def errors(z, estimated):
    r = np.abs(estimated / z - 1)
    return {"near_absrel_pct": float(100*r[(z >= .3) & (z <= 4)].mean()),
            "far_absrel_pct": float(100*r[z >= 12].mean()), "grid_max_rel_pct": float(100*r.max())}


def normalized(x, left, right):
    return (x-left)/(right-left)


def finite_cells(encode, inverse, left, right, count):
    # Monotone inverse: one-sided limiting errors at quantizer midpoints suffice.
    u = np.linspace(left, right, count)
    cuts = inverse((u[:-1] + u[1:]) / 2)
    recon = inverse(u)
    return float(100 * max(np.max(abs(recon[:-1]/cuts-1)), np.max(abs(recon[1:]/cuts-1))))


def run():
    check_symbolic()
    z = np.geomspace(LO, HI, 200001)
    rng = np.random.default_rng(73041)
    mixed = mixed_primitive(z)
    maps = {"depth": (lambda x:x, lambda x:x), "disp": (np.reciprocal, np.reciprocal),
            "log": (np.log, np.exp), "sqrt": (np.sqrt, np.square),
            "mixed_minimax": (mixed_primitive, lambda x:np.interp(x, mixed, z))}
    results = {}
    for name, (enc, inv) in maps.items():
        left, right = enc(LO), enc(HI)
        u = normalized(enc(z), left, right)
        result = {}
        for sym, n in (("affine_uint8", 256), ("symmetric_int8", 255)):
            q = (np.rint(127*(2*u-1)).clip(-127,127)/127+1)/2 if n == 255 else np.rint(255*u).clip(0,255)/255
            stats = errors(z, inv(left + q*(right-left)))
            stats["cell_sup_rel_pct"] = finite_cells(enc, inv, left, right, n)
            result[sym] = stats
        result["fp16_native_coordinate"] = errors(z, inv(enc(z).astype(np.float16).astype(float)))
        # Prescribe [1,256] and integrate actual normal binade density, exactly
        # 1024 intervals per exponent. Equal spacing in normalized length u.
        nbin = 8*u
        exponent = np.floor(nbin).astype(int)
        hw = np.ldexp(1 + nbin - exponent, exponent).astype(np.float16).astype(float)
        mant, exponent = np.frexp(hw)
        u_recon = (exponent-1 + (2*mant-1))/8
        result["fp16_equal_binade_mass_1_256"] = errors(z, inv(left+u_recon*(right-left)))
        results[name] = result

    smax = np.tanh(HALF_SPAN/2)
    q = smax/(1+np.sqrt(1-smax**2))
    e_enc = 8*smax/np.sqrt(1-smax**2)*q**32/(1-q)
    h = (HALF_SPAN + e_enc)/32
    e_dec = 32*h**5/(5*(144-12*h*h))
    # Includes the rational encoder and exact-real decoder roundtrip without quantization.
    z_mc = np.exp(rng.uniform(np.log(LO), np.log(HI), 200000))
    observed = np.max(abs(decode(logcode(1/z_mc))-z_mc))
    interval_bound = HI * np.expm1(e_enc + e_dec)
    assert observed <= interval_bound < 1e-4
    u = 2**-11
    fp16_bound = 2*u/(1-u)
    isolated = (1/z).astype(np.float16)
    rounded = np.reciprocal(isolated.astype(float)).astype(np.float16).astype(float).clip(LO,HI)
    fp16_stats = errors(z, rounded)
    assert fp16_stats["grid_max_rel_pct"] <= 100*fp16_bound < 0.15
    by_name = {k:v["affine_uint8"]["cell_sup_rel_pct"] for k,v in results.items()}
    # Bound on quantizer decision shift, Pade decode, and one code half-width.
    log_impl_sup = 100 * np.expm1(HALF_SPAN/255 + e_enc + e_dec)
    assert by_name["disp"] / log_impl_sup >= 5 and by_name["depth"] / log_impl_sup >= 8
    g = mixed[-1] - mixed[0]
    w = lambda x: 1/x**2 + 1/x**4
    integral, _ = quad(lambda x:w(x)**(1/3), LO, HI, epsabs=1e-11)
    improvement = {}
    for name, span, jac, max_at in (("depth",HI-LO,lambda x:1,LO),
                                    ("disp",1/LO-1/HI,lambda x:1/x**2,HI),
                                    ("log",2*HALF_SPAN,lambda x:1/x,LO)):
        d, _ = quad(lambda x, derivative=jac: w(x)/derivative(x)**2, LO, HI)
        improvement[name] = {"minimax_weighted_width_ratio":span*np.sqrt(w(max_at))/jac(max_at)/g,
                             "uniform_source_mse_ratio_vs_expected_optimum":span**2*d/integral**3}
    return {"range_m":[LO,HI], "samples_log_uniform":len(z), "mc_seed":73041, "mc_size":len(z_mc),
            "symbolic_checks_passed":True, "design_beta":[1,1,0],
            "near_m":[.3,4], "far_m":[12,25], "metrics":results,
            "mixed_risk_high_rate":improvement, "encoder_log_abs_bound":e_enc,
            "decoder_log_abs_bound":e_dec, "real_roundtrip_abs_bound_m":interval_bound,
            "mc_roundtrip_max_abs_m":float(observed),
            "implemented_affine_int8_rel_sup_upper_pct":log_impl_sup,
            "affine_int8_improvement_lower_bound":{k:by_name[k]/log_impl_sup for k in ("depth","disp")},
            "fp16_disp_plus_reciprocal":fp16_stats, "fp16_disp_plus_reciprocal_bound_pct":100*fp16_bound}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    report = run()
    (OUT/"theory_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "metrics"}, indent=2))
