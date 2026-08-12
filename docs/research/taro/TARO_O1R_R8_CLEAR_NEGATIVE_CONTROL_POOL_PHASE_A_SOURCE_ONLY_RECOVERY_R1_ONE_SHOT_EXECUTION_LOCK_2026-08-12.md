# TARO O1R R8 Phase-A source-only recovery R1 execution lock

Status: `AUTHORIZED_UNCONSUMED`.

The interrupted R0 root contains all 402 sealed DepthART candidate inputs,
records, and native-depth blobs. Before this lock was frozen, every candidate
was replayed and its blob, array, high-resolution depth, record sequence, and
input sequence were verified against the sealed candidate completion.

R1 adopts those candidates into a fresh root and reruns only Apple
depth/confidence source-feature construction. It does not adopt the partial R0
source records and performs zero model inference, FARO/truth read, threshold
fit, training, or network request.
