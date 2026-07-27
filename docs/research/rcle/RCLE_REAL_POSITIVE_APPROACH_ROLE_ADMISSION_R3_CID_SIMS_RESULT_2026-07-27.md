# RCLE R3 CID-SIMS result

## Outcome

Useful real positive-approach data was found in the official CID-SIMS V6
`floor3_1` archive.

The formal R3 evidence is **invalid**, because its frozen ZIP wrapper required
`floor3_1/pose.txt`, while the official archive actually contains
`floor3_1/groundtruth.txt`. The formal attempt stopped at central-directory
metadata and did not evaluate geometry.

The same consumed claim was then used for an explicitly non-formal,
geometry-only diagnostic. The diagnostic changed only the observed control
filename. It retained the pre-access archive identity, fixed non-overlapping
ten-second partition, deterministic 24-pair sampling, geometry implementation,
and numerical gates.

## Verified source identity

- official file ID: `c595882daafe788a29d687872cc1fc2a`
- bytes: `2,211,008,069`
- official MD5: `585d38855ad7d04817991cdbbb72016b`
- local SHA-256:
  `b622be7918d0003c97f0e33cc30071c9995f49c59726240e7475f2cde8572984`
- exact shared RGB-depth timestamps: `3,834`
- pose rows in `groundtruth.txt`: `12,823`

The archive was byte- and MD5-verified before it was opened. No RGB pixels or
RGB-algorithm outcomes were read.

## Geometry finding

All 12 complete, fixed, non-overlapping ten-second windows were evaluated.
Eleven passed. The first window missed only the signed-radial threshold.

The deterministic selected development-canary window is the earliest passing
window:

- zero-based window index: `1`
- half-open interval:
  `[1673419232.281298, 1673419242.281298)`
- evaluable pairs: `24 / 24`
- coverage: `1.0`
- median signed radial expansion: `0.4435781894989603 / s`
- median positive fraction: `0.9978479252143693`
- median q90 time-normalized parallax: `0.5097600314373266 rad / s`

The independent result-and-ledger aggregation validation returned `valid=true`,
no errors, and independently selected window index `1`.

## Authority

This establishes that `floor3_1` contains genuinely useful real RGB-D plus pose
data for a **geometry-selected development canary**. It does not rescue the
formal R3 admission, and it does not create confirmation, performance,
product, or safety authority.

The next executable task may bind the exact archive and selected window for a
separate RGB-algorithm development-canary run. It must not describe that run as
independent confirmation or product qualification.

## Evidence hashes

- geometry claim:
  `b4dc2d16e759d979a941b9ab20c40388b3de067ba23bdea2ce04a473f51d2e96`
- diagnostic result:
  `db4fa851615750b46e56d34086681cda6181866f82aefa0348f57ae4f02e34c1`
- pair ledger:
  `4610f1477435731031e1610d2b710fdc16f7c8dfa9e336583893e3c505b0752b`
- independent validation:
  `8662a9f3b58e1d84bb2305b45cb54d6e44aa969c83a73c242f10dae80ea2a373`
