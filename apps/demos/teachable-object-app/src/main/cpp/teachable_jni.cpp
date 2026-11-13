// Teachable personal-object recognition, native side:
//  - a generic SNPE engine (DLC, CPU or HTP runtime, any number of float inputs/outputs);
//  - the discriminative local memory build (Graduation thesis main method, orbit_dump.part_memory_from) in FP32 on the CPU;
//  - torchvision roi_align(aligned=True, sampling_ratio=2) for the on-demand second look.
// Random sampling uses splitmix64 + partial Fisher-Yates so the PC reference can reproduce it; it is not bit-identical
// to PyTorch's randperm.
#include <jni.h>
#include <arm_neon.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "DlContainer/DlContainer.h"
#include "DlSystem/DlError.h"
#include "DlSystem/ITensor.h"
#include "DlSystem/PlatformConfig.h"
#include "DlSystem/RuntimeList.h"
#include "DlSystem/StringList.h"
#include "DlSystem/TensorMap.h"
#include "DlSystem/TensorShape.h"
#include "SNPE/SNPE.h"
#include "SNPE/SNPEBuilder.h"
#include "SNPE/SNPEUtil.h"

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        const char* detail = Snpe_ErrorCode_GetLastErrorString();
        throw std::runtime_error(std::string(message) + ": " + (detail ? detail : "no SDK detail"));
    }
}
template<auto Delete> struct Owned {
    void* value = nullptr;
    ~Owned() { if (value) Delete(value); }
};
struct Engine {
    Snpe_SNPE_Handle_t snpe = nullptr;
    std::vector<std::string> inputs, outputs;
    std::vector<Snpe_ITensor_Handle_t> tensors;
    ~Engine() {
        for (auto t : tensors) Snpe_ITensor_Delete(t);
        if (snpe) Snpe_SNPE_Delete(snpe);
    }
};
void fail(JNIEnv* env, const std::exception& error) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error.what());
}
std::string string(JNIEnv* env, jstring value) {
    const char* p = env->GetStringUTFChars(value, nullptr);
    if (!p) throw std::runtime_error("Unable to read JNI string");
    std::string result(p);
    env->ReleaseStringUTFChars(value, p);
    return result;
}
std::vector<std::string> names(Snpe_StringList_Handle_t list) {
    std::vector<std::string> out;
    for (size_t i = 0; i < Snpe_StringList_Size(list); ++i) out.emplace_back(Snpe_StringList_At(list, i));
    return out;
}

// ---- memory build ----------------------------------------------------------------------------------------------------
// Feature width: 384 for DINOv2-S, 768 for DINOv2-B. JNI serializes builds while the worker threads use this width.
static int D = 384;
std::mutex memory_build_lock;

struct Rng {
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
};
// First k entries of a uniformly random permutation of items (partial Fisher-Yates).
std::vector<int> sample(std::vector<int> items, int k, uint64_t seed) {
    Rng rng{seed};
    k = std::min<int>(k, items.size());
    for (int i = 0; i < k; ++i) std::swap(items[i], items[i + rng.next() % (items.size() - i)]);
    items.resize(k);
    return items;
}

inline float dot(const float* a, const float* b) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    for (int k = 0; k < D; k += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + k), vld1q_f32(b + k));
        s1 = vfmaq_f32(s1, vld1q_f32(a + k + 4), vld1q_f32(b + k + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + k + 8), vld1q_f32(b + k + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + k + 12), vld1q_f32(b + k + 12));
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
}

constexpr int max_memory_threads = 8;

int memory_threads() {
    return std::max(1u, std::min(unsigned(max_memory_threads), std::thread::hardware_concurrency()));
}

std::vector<float> pack_columns(const float* features, const std::vector<int>& columns) {
    std::vector<float> packed(columns.size() * D);
    for (size_t i = 0; i < columns.size(); ++i) {
        std::copy_n(features + size_t(columns[i]) * D, D, packed.data() + i * D);
    }
    return packed;
}

template<class Work> void memory_workers(int threads, Work work) {
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back(work, t);
    for (auto& worker : pool) worker.join();
}

// For every row r of A (rows x D) visit dot(A_r, B_c) for all columns c of B (cols x D, gathered by index), tiled so a
// block of B stays in cache; visit(row, col, value) must be thread-safe per row. Rows are split over threads.
template<class Visit> void all_dots(const float* A, int rows, const float* B, const std::vector<int>& cols, Visit visit) {
    const int threads = memory_threads();
    constexpr int tile = 256;
    const auto packed = pack_columns(B, cols);
    memory_workers(threads, [&](int t) {
        const int lo = rows * t / threads, hi = rows * (t + 1) / threads;
        for (size_t c0 = 0; c0 < cols.size(); c0 += tile) {
            const size_t c1 = std::min(cols.size(), c0 + tile);
            for (int r = lo; r < hi; ++r)
                for (size_t c = c0; c < c1; ++c) visit(r, int(c), dot(A + size_t(r) * D, packed.data() + c * D));
        }
    });
}

void normalize(float* v) {
    float n = std::sqrt(dot(v, v));
    n = std::max(n, 1e-12f);
    for (int k = 0; k < D; ++k) v[k] /= n;
}

// orbit_eval.kmeans: init from a random subset, 15 rounds of argmax assignment + normalised means.
std::vector<float> kmeans(const std::vector<float>& x, int n, int k, uint64_t seed) {
    std::vector<int> all(n); std::iota(all.begin(), all.end(), 0);
    auto init = sample(all, k, seed);
    std::vector<float> c(size_t(k) * D);
    for (int j = 0; j < k; ++j) std::copy_n(x.data() + size_t(init[j]) * D, D, c.data() + size_t(j) * D);
    std::vector<int> assign(n);
    std::vector<float> best(n);
    std::vector<int> cidx(k); std::iota(cidx.begin(), cidx.end(), 0);
    for (int it = 0; it < 15; ++it) {
        std::fill(best.begin(), best.end(), -INFINITY);
        all_dots(x.data(), n, c.data(), cidx, [&](int r, int j, float v) { if (v > best[r]) { best[r] = v; assign[r] = j; } });
        for (int j = 0; j < k; ++j) {
            std::vector<double> sum(D, 0.0); int count = 0;
            for (int r = 0; r < n; ++r) if (assign[r] == j) { ++count; for (int d = 0; d < D; ++d) sum[d] += x[size_t(r) * D + d]; }
            if (!count) continue;
            for (int d = 0; d < D; ++d) c[size_t(j) * D + d] = float(sum[d] / count);
            normalize(c.data() + size_t(j) * D);
        }
    }
    return c;
}

// part_memory_from (non-colour, own term on): per object, s_other = max similarity to <= 30,000 sampled patches of the
// user's other objects, s_own = mean of the 2nd..6th best similarity to <= 30,000 own patches; keep d = s_own - s_other
// at or above its (1 - keep) quantile (linear interpolation, as torch.quantile); k-means into <= 32 parts. Output is
// padded to 32 parts per object by repeating parts (the max over parts is unchanged).
// other_max(P, n, all, others, s_other): s_other[r] = max_c P_r . all[others[c]] (CPU or HTP). ms[3] receives the time
// spent in the other-object max, the own top-6 and k-means.
using OtherMax = std::function<void(const float*, int, const float*, const std::vector<int>&, std::vector<float>&)>;

void cpu_other_max(const float* P, int n, const float* all, const std::vector<int>& others, std::vector<float>& s) {
    all_dots(P, n, all, others, [&](int r, int, float v) { if (v > s[r]) s[r] = v; });
}

using TopSix = std::array<float, 6>;

float own_score(const TopSix& top) {
    return (top[1] + top[2] + top[3] + top[4] + top[5]) / 5.f;
}

double quantile_position(int n, float keep) {
    return (n - 1) * (1.0 - keep);
}

struct MemoryDiagnostics {
    jlong negative_dots = 0;
    jlong positive_dots = 0;
    jlong pruned_rows = 0;
    jlong reused_witness_rows = 0;
    jlong ineligible_witness_rows = 0;
    jlong covered_negative_dots = 0;
};

struct TeachingCache {
    int objects = 0;
    // Per row: exact own score and maximum negative score over the covered prefix of its sampled negatives.
    std::vector<float> scores;
    // Per row: global index attaining that maximum and the length of the covered prefix.
    std::vector<jint> coverage;
    // Actual negative traversal order, packed in class order with one slice per sampled set.
    std::vector<jint> order;
};

constexpr int patch_sample_cap = 30000;

int negative_sample_count(int total_rows, int own_rows) {
    return std::min(patch_sample_cap, total_rows - own_rows);
}

int negative_order_size(const std::vector<int>& counts, int objects, int rows) {
    int size = 0;
    for (int o = 0; o < objects; ++o) size += negative_sample_count(rows, counts[o]);
    return size;
}

std::vector<int> sample_negatives(const std::vector<int>& offsets, int objects, int object) {
    std::vector<int> others;
    for (int j = 0; j < objects; ++j) {
        if (j != object) {
            for (int i = offsets[j]; i < offsets[j + 1]; ++i) others.push_back(i);
        }
    }
    return sample(std::move(others), patch_sample_cap, 2ull * object);
}

std::vector<int> prepare_negative_reuse(std::vector<int>& others, const std::vector<int>& offsets, int object,
                                        const TeachingCache& cache, int order_offset, std::vector<float>& maxima,
                                        MemoryDiagnostics& diagnostics, jint* next_coverage) {
    const int old_rows = offsets[cache.objects];
    const int old_size = negative_sample_count(old_rows, offsets[object + 1] - offsets[object]);
    const jint* old_order = cache.order.data() + order_offset;
    std::vector<int> old_position(old_rows, -1);
    for (int i = 0; i < old_size; ++i) {
        const int col = old_order[i];
        require(col >= 0 && col < old_rows && old_position[col] < 0, "Invalid cached negative order");
        old_position[col] = i;
    }
    std::vector<uint8_t> in_new_sample(offsets.back());
    for (int col : others) in_new_sample[col] = 1;
    std::vector<int> ordered;
    ordered.reserve(others.size());

    // Visit the intersection in old order, then newly sampled columns. A surviving covered prefix stays a prefix.
    std::vector<int> surviving_prefix(old_size + 1);
    for (int i = 0; i < old_size; ++i) {
        const int col = old_order[i];
        if (in_new_sample[col]) ordered.push_back(col);
        surviving_prefix[i + 1] = static_cast<int>(ordered.size());
    }
    for (int col : others) if (col >= old_rows || old_position[col] < 0) ordered.push_back(col);
    others = std::move(ordered);

    std::vector<int> covered(maxima.size());
    for (size_t r = 0; r < maxima.size(); ++r) {
        const size_t i = 2 * (size_t(offsets[object]) + r);
        const int witness = cache.coverage[i];
        const int checked = cache.coverage[i + 1];
        if (witness >= 0 && witness < old_rows && in_new_sample[witness] && old_position[witness] >= 0 &&
            old_position[witness] < checked && checked <= old_size) {
            maxima[r] = cache.scores[i + 1];
            covered[r] = surviving_prefix[checked];
            if (next_coverage) {
                next_coverage[2 * r] = witness;
                next_coverage[2 * r + 1] = covered[r];
            }
            ++diagnostics.reused_witness_rows;
        } else {
            ++diagnostics.ineligible_witness_rows;
        }
    }
    return covered;
}

// A partial negative maximum gives an upper bound on the final margin. The k-th highest complete margin is a lower
// bound on the lower endpoint of the target quantile. Rows strictly below it may keep their partial maximum.
template<bool Capture = false, bool Reuse = false>
void cutoff_other_max(const float* queries, const float* all, const std::vector<int>& others,
                      const std::vector<float>& own, float keep, std::vector<float>& maxima,
                      MemoryDiagnostics& diagnostics, jint* coverage = nullptr, const int* covered = nullptr) {
    const int n = static_cast<int>(own.size());
    const int columns = static_cast<int>(others.size());
    const int prefix = std::min(256, columns);
    const int threads = memory_threads();
    constexpr int tile = 128;
    constexpr int row_batch = 512;
    const auto packed = pack_columns(all, others);
    std::vector<float> upper(n);
    auto accumulate = [&](int row, int first, int last) {
        const float* query = queries + size_t(row) * D;
        float maximum = maxima[row];
        int witness = -1;
        if constexpr (Capture) witness = coverage[2 * row];
        int begin = first;
        if constexpr (Reuse) begin = std::clamp(covered[row], first, last);
        for (int col = begin; col < last; ++col) {
            const float similarity = dot(query, packed.data() + size_t(col) * D);
            if (similarity > maximum) {
                maximum = similarity;
                if constexpr (Capture) witness = others[col];
            }
        }
        maxima[row] = maximum;
        if constexpr (Capture) {
            coverage[2 * row] = witness;
            coverage[2 * row + 1] = std::max(coverage[2 * row + 1], last);
        }
        return begin - first;
    };
    std::array<jlong, max_memory_threads> prefix_skips{};
    memory_workers(threads, [&](int t) {
        const int lo = n * t / threads, hi = n * (t + 1) / threads;
        for (int first = 0; first < prefix; first += tile) {
            const int last = std::min(first + tile, prefix);
            for (int row = lo; row < hi; ++row) prefix_skips[t] += accumulate(row, first, last);
        }
        for (int row = lo; row < hi; ++row) upper[row] = own[row] - maxima[row];
    });
    const jlong prefix_skipped = std::accumulate(prefix_skips.begin(), prefix_skips.end(), jlong{0});
    diagnostics.covered_negative_dots += prefix_skipped;
    diagnostics.negative_dots += jlong(n) * prefix - prefix_skipped;
    if (prefix == columns) return;

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return upper[a] > upper[b]; });
    const int k = n - static_cast<int>(std::floor(quantile_position(n, keep)));
    std::priority_queue<float, std::vector<float>, std::greater<float>> best;
    std::vector<uint8_t> complete(n);
    for (int lo = 0; lo < n;) {
        const float cutoff = static_cast<int>(best.size()) == k ? best.top() : -INFINITY;
        if (upper[order[lo]] < cutoff) break;
        const int hi = std::min(lo + (lo == 0 ? k : row_batch), n);
        std::array<jlong, max_memory_threads> dots{}, skips{};
        memory_workers(threads, [&](int t) {
            std::vector<int> active;
            for (int i = lo + t; i < hi && upper[order[i]] >= cutoff; i += threads) active.push_back(order[i]);
            for (int first = prefix; first < columns && !active.empty(); first += tile) {
                const int last = std::min(first + tile, columns);
                dots[t] += jlong(active.size()) * (last - first);
                size_t survivors = 0;
                for (int row : active) {
                    skips[t] += accumulate(row, first, last);
                    if (last == columns) complete[row] = 1;
                    else if (own[row] - maxima[row] >= cutoff) active[survivors++] = row;
                }
                active.resize(survivors);
            }
        });
        const jlong skipped = std::accumulate(skips.begin(), skips.end(), jlong{0});
        diagnostics.covered_negative_dots += skipped;
        diagnostics.negative_dots += std::accumulate(dots.begin(), dots.end(), jlong{0}) - skipped;
        for (int i = lo; i < hi; ++i) {
            const int row = order[i];
            if (!complete[row]) continue;
            const float score = own[row] - maxima[row];
            if (static_cast<int>(best.size()) < k) best.push(score);
            else if (score > best.top()) {
                best.pop();
                best.push(score);
            }
        }
        lo = hi;
    }
    diagnostics.pruned_rows += std::count(complete.begin(), complete.end(), 0);
}

// HTP simmax graph a [rows, D], b [cols, D] -> mx [rows]. The other-object patches are split into column blocks of
// `cols` (outer loop: each block uploaded once, the last one padded by repeating its first row, max unchanged); the
// object's patches are streamed through a in row chunks (last chunk zero-padded, outputs ignored); s keeps the running
// max. Short blocks matter: on the SM8650 HTP a 1024 x 7500 call runs about 7x more efficiently per MAC than 1024 x 30000.
void htp_other_max(Engine* e, const float* P, int n, const float* all, const std::vector<int>& others, std::vector<float>& s) {
    const size_t ia = std::find(e->inputs.begin(), e->inputs.end(), "a") - e->inputs.begin();
    const size_t ib = std::find(e->inputs.begin(), e->inputs.end(), "b") - e->inputs.begin();
    require(ia < e->inputs.size() && ib < e->inputs.size(), "simmax inputs a, b");
    const size_t rows = Snpe_ITensor_GetSize(e->tensors[ia]) / D, cols = Snpe_ITensor_GetSize(e->tensors[ib]) / D;
    auto* A = static_cast<float*>(Snpe_ITensor_GetData(e->tensors[ia]));
    auto* B = static_cast<float*>(Snpe_ITensor_GetData(e->tensors[ib]));
    Owned<Snpe_TensorMap_Delete> in{Snpe_TensorMap_Create()};
    Snpe_TensorMap_Add(in.value, "a", e->tensors[ia]); Snpe_TensorMap_Add(in.value, "b", e->tensors[ib]);
    for (size_t c0 = 0; c0 < others.size(); c0 += cols) {
        for (size_t c = 0; c < cols; ++c) {
            const size_t k = c0 + c < others.size() ? c0 + c : c0;
            std::copy_n(all + size_t(others[k]) * D, D, B + c * D);
        }
        for (int r0 = 0; r0 < n; r0 += int(rows)) {
            const int m = std::min<int>(rows, n - r0);
            std::copy_n(P + size_t(r0) * D, size_t(m) * D, A);
            std::fill(A + size_t(m) * D, A + rows * D, 0.f);
            Owned<Snpe_TensorMap_Delete> out{Snpe_TensorMap_Create()};
            require(Snpe_SNPE_ExecuteITensors(e->snpe, in.value, out.value) == SNPE_SUCCESS, "Execute simmax");
            auto t = Snpe_TensorMap_GetTensor_Ref(out.value, "mx");
            require(t && Snpe_ITensor_GetSize(t) == rows, "simmax output");
            const float* mx = static_cast<const float*>(Snpe_ITensor_GetData(t));
            for (int i = 0; i < m; ++i) s[r0 + i] = std::max(s[r0 + i], mx[i]);
        }
    }
}

// HTP simtop6 graph a [rows, D], b [cols, D] -> tv [rows, 6] (descending). Same blocking as htp_other_max, but b blocks
// are padded with zero vectors (similarity 0), not repeated rows, which would duplicate values in a top-6; each
// block's top-6 is merged into the row's running top-6. On the SM8650 HTP TopK limits this to ~60-80 GMAC/s at any
// tested width; 1024 x 2560 was the best trade-off.
void htp_own_top6(Engine* e, const float* P, int n, const float* all, const std::vector<int>& own,
                  std::vector<TopSix>& top) {
    const size_t ia = std::find(e->inputs.begin(), e->inputs.end(), "a") - e->inputs.begin();
    const size_t ib = std::find(e->inputs.begin(), e->inputs.end(), "b") - e->inputs.begin();
    require(ia < e->inputs.size() && ib < e->inputs.size(), "simtop inputs a, b");
    const size_t rows = Snpe_ITensor_GetSize(e->tensors[ia]) / D, cols = Snpe_ITensor_GetSize(e->tensors[ib]) / D;
    auto* A = static_cast<float*>(Snpe_ITensor_GetData(e->tensors[ia]));
    auto* B = static_cast<float*>(Snpe_ITensor_GetData(e->tensors[ib]));
    Owned<Snpe_TensorMap_Delete> in{Snpe_TensorMap_Create()};
    Snpe_TensorMap_Add(in.value, "a", e->tensors[ia]); Snpe_TensorMap_Add(in.value, "b", e->tensors[ib]);
    for (size_t c0 = 0; c0 < own.size(); c0 += cols) {
        for (size_t c = 0; c < cols; ++c) {
            if (c0 + c < own.size()) std::copy_n(all + size_t(own[c0 + c]) * D, D, B + c * D);
            else std::fill(B + c * D, B + (c + 1) * D, 0.f);
        }
        for (int r0 = 0; r0 < n; r0 += int(rows)) {
            const int m = std::min<int>(rows, n - r0);
            std::copy_n(P + size_t(r0) * D, size_t(m) * D, A);
            std::fill(A + size_t(m) * D, A + rows * D, 0.f);
            Owned<Snpe_TensorMap_Delete> out{Snpe_TensorMap_Create()};
            require(Snpe_SNPE_ExecuteITensors(e->snpe, in.value, out.value) == SNPE_SUCCESS, "Execute simtop");
            auto t = Snpe_TensorMap_GetTensor_Ref(out.value, "tv");
            require(t && Snpe_ITensor_GetSize(t) == rows * 6, "simtop output");
            const float* tv = static_cast<const float*>(Snpe_ITensor_GetData(t));
            for (int i = 0; i < m; ++i) {
                auto& tr = top[r0 + i];
                for (int k = 0; k < 6; ++k) {
                    const float v = tv[i * 6 + k];
                    if (v <= tr[5]) break;                                  // block values are descending
                    int j = 5; while (j > 0 && tr[j - 1] < v) { tr[j] = tr[j - 1]; --j; } tr[j] = v;
                }
            }
        }
    }
}

double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

std::vector<float> build_memory(const std::vector<float>& patches, const std::vector<int>& counts, float keep, int parts,
                                const OtherMax& other_max, double* ms, Engine* simtop, bool cutoff,
                                std::vector<jint>* retained, MemoryDiagnostics& diagnostics,
                                const TeachingCache& cache, TeachingCache* next_cache) {
    const int O = counts.size();
    std::vector<int> offs(O + 1, 0);
    for (int o = 0; o < O; ++o) offs[o + 1] = offs[o] + counts[o];
    std::vector<float> out(size_t(O) * parts * D);
    // phase 1: other-object max for every object; phase 2: own top-6 for every object. Running each graph over all
    // objects before switching avoids alternating two HTP graphs per object. The CPU cutoff needs top-6 first.
    std::vector<std::vector<float>> s_others(O);
    std::vector<std::vector<float>> s_owns(O);
    auto negatives = [&] {
        const auto t = std::chrono::steady_clock::now();
        int old_order_offset = 0, next_order_offset = 0;
        for (int o = 0; o < O; ++o) {
            auto others = sample_negatives(offs, O, o);
            auto& maxima = s_others[o];
            maxima.assign(counts[o], -INFINITY);
            const float* query = patches.data() + size_t(offs[o]) * D;
            const size_t score_offset = 2 * size_t(offs[o]);
            auto* next_coverage = next_cache ? next_cache->coverage.data() + score_offset : nullptr;
            if (o < cache.objects) {
                const auto covered = prepare_negative_reuse(others, offs, o, cache, old_order_offset, maxima,
                                                            diagnostics, next_coverage);
                old_order_offset += negative_sample_count(offs[cache.objects], counts[o]);
                if (next_cache) {
                    cutoff_other_max<true, true>(query, patches.data(), others, s_owns[o], keep, maxima, diagnostics,
                                                next_coverage, covered.data());
                } else {
                    cutoff_other_max<false, true>(query, patches.data(), others, s_owns[o], keep, maxima, diagnostics,
                                                 nullptr, covered.data());
                }
            } else if (next_cache) {
                cutoff_other_max<true>(query, patches.data(), others, s_owns[o], keep, maxima, diagnostics, next_coverage);
            } else if (cutoff) {
                cutoff_other_max(query, patches.data(), others, s_owns[o], keep, maxima, diagnostics);
            } else {
                other_max(query, counts[o], patches.data(), others, maxima);
                diagnostics.negative_dots += jlong(counts[o]) * others.size();
            }
            if (next_cache) {
                std::copy(others.begin(), others.end(), next_cache->order.data() + next_order_offset);
                next_order_offset += static_cast<int>(others.size());
                for (int r = 0; r < counts[o]; ++r) {
                    next_cache->scores[score_offset + 2 * r] = s_owns[o][r];
                    next_cache->scores[score_offset + 2 * r + 1] = maxima[r];
                }
            }
        }
        ms[0] += since(t);
    };
    if (!cutoff) negatives();
    auto t = std::chrono::steady_clock::now();
    for (int o = 0; o < O; ++o) {
        const int n = counts[o];
        auto& scores = s_owns[o];
        scores.resize(n);
        if (o < cache.objects) {
            for (int r = 0; r < n; ++r) scores[r] = cache.scores[2 * (size_t(offs[o]) + r)];
            continue;
        }
        const float* P = patches.data() + size_t(offs[o]) * D;
        std::vector<int> own(n); std::iota(own.begin(), own.end(), offs[o]);
        own = sample(std::move(own), patch_sample_cap, 2ull * o + 1);
        std::vector<TopSix> top(n);
        for (auto& tr : top) tr.fill(-INFINITY);
        if (simtop) htp_own_top6(simtop, P, n, patches.data(), own, top);
        else all_dots(P, n, patches.data(), own, [&](int r, int, float v) {
            auto& tr = top[r];
            if (v <= tr[5]) return;
            int i = 5; while (i > 0 && tr[i - 1] < v) { tr[i] = tr[i - 1]; --i; } tr[i] = v;
        });
        std::transform(top.begin(), top.end(), scores.begin(), own_score);
        diagnostics.positive_dots += jlong(n) * own.size();
    }
    ms[1] += since(t);
    if (cutoff) negatives();
    for (int o = 0; o < O; ++o) {
        t = std::chrono::steady_clock::now();
        const int n = counts[o];
        const float* P = patches.data() + size_t(offs[o]) * D;
        const auto& s_other = s_others[o]; const auto& s_own = s_owns[o];
        std::vector<float> d(n);
        for (int r = 0; r < n; ++r) d[r] = s_own[r] - s_other[r];
        std::vector<float> sorted = d; std::sort(sorted.begin(), sorted.end());
        const double pos = quantile_position(n, keep); const int lo = int(std::floor(pos)); const int hi = std::min(lo + 1, n - 1);
        const float q = float(sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]));
        std::vector<float> kept;
        for (int r = 0; r < n; ++r) if (d[r] >= q) {
            kept.insert(kept.end(), P + size_t(r) * D, P + size_t(r + 1) * D);
            if (retained) retained->push_back(offs[o] + r);
        }
        const int nk = kept.size() / D, k = std::min(parts, nk);
        auto c = kmeans(kept, nk, k, 0);
        for (int j = 0; j < parts; ++j) std::copy_n(c.data() + size_t(j % k) * D, D, out.data() + (size_t(o) * parts + j) * D);
        ms[2] += since(t);
    }
    return out;
}

// torchvision.ops.roi_align, aligned=True, sampling_ratio=2, spatial_scale=1, one box, CHW float image.
std::vector<float> roi_align(const float* img, int C, int H, int W, float x0, float y0, float x1, float y1, int out) {
    std::vector<float> res(size_t(C) * out * out);
    const float sx = x0 - .5f, sy = y0 - .5f, bw = (x1 - x0) / out, bh = (y1 - y0) / out;
    for (int ph = 0; ph < out; ++ph) for (int pw = 0; pw < out; ++pw) {
        float acc[3] = {0, 0, 0};
        for (int iy = 0; iy < 2; ++iy) for (int ix = 0; ix < 2; ++ix) {
            float y = sy + ph * bh + (iy + .5f) * bh / 2, x = sx + pw * bw + (ix + .5f) * bw / 2;
            if (y < -1.f || y > H || x < -1.f || x > W) continue;
            y = std::max(y, 0.f); x = std::max(x, 0.f);
            int yl = int(y), xl = int(x), yh, xh;
            if (yl >= H - 1) { yh = yl = H - 1; y = float(yl); } else yh = yl + 1;
            if (xl >= W - 1) { xh = xl = W - 1; x = float(xl); } else xh = xl + 1;
            const float ly = y - yl, lx = x - xl, hy = 1 - ly, hx = 1 - lx;
            for (int ch = 0; ch < C; ++ch) {
                const float* p = img + size_t(ch) * H * W;
                acc[ch] += hy * hx * p[yl * W + xl] + hy * lx * p[yl * W + xh] + ly * hx * p[yh * W + xl] + ly * lx * p[yh * W + xh];
            }
        }
        for (int ch = 0; ch < C; ++ch) res[(size_t(ch) * out + ph) * out + pw] = acc[ch] / 4.f;
    }
    return res;
}
}  // namespace

#define JFN(name) Java_com_linnan_blindassist_teachable_Native_##name

// output_tensors: names to return; empty = SNPE's default (the last layer only).
extern "C" JNIEXPORT jlong JNICALL JFN(engineCreate)(JNIEnv* env, jobject, jstring path, jboolean htp, jstring dsp_path,
                                                     jobjectArray output_tensors) {
    try {
        if (htp) {
            const auto dsp = string(env, dsp_path);
            const char* previous = std::getenv("ADSP_LIBRARY_PATH");
            std::string search = dsp + ";" + (previous ? previous : "/vendor/lib/rfsa/adsp;/vendor/dsp;/system/lib/rfsa/adsp;/dsp");
            require(setenv("ADSP_LIBRARY_PATH", search.c_str(), 1) == 0, "Set DSP search path");
            require(Snpe_Util_IsRuntimeAvailableCheckOption(SNPE_RUNTIME_DSP, SNPE_RUNTIME_CHECK_OPTION_UNSIGNEDPD_CHECK), "HTP runtime unavailable");
        }
        Owned<Snpe_DlContainer_Delete> container{Snpe_DlContainer_Open(string(env, path).c_str())};
        require(container.value, "Open DLC");
        Owned<Snpe_SNPEBuilder_Delete> builder{Snpe_SNPEBuilder_Create(container.value)};
        Owned<Snpe_RuntimeList_Delete> runtimes{Snpe_RuntimeList_Create()};
        Owned<Snpe_PlatformConfig_Delete> platform{Snpe_PlatformConfig_Create()};
        require(builder.value && runtimes.value && platform.value, "Create SNPE builder");
        require(Snpe_RuntimeList_Add(runtimes.value, htp ? SNPE_RUNTIME_DSP : SNPE_RUNTIME_CPU) == SNPE_SUCCESS, "Runtime order");
        if (htp) require(Snpe_PlatformConfig_SetPlatformOptions(platform.value, "unsignedPD:ON"), "Unsigned DSP platform");
        require(Snpe_SNPEBuilder_SetRuntimeProcessorOrder(builder.value, runtimes.value) == SNPE_SUCCESS, "Set runtime");
        require(Snpe_SNPEBuilder_SetPlatformConfig(builder.value, platform.value) == SNPE_SUCCESS, "Set platform");
        require(Snpe_SNPEBuilder_SetPerformanceProfile(builder.value, SNPE_PERFORMANCE_PROFILE_BURST) == SNPE_SUCCESS, "Set burst");
        require(Snpe_SNPEBuilder_SetUseUserSuppliedBuffers(builder.value, false) == SNPE_SUCCESS, "Set ITensor");
        Owned<Snpe_StringList_Delete> wanted{Snpe_StringList_Create()};
        for (jsize i = 0; i < env->GetArrayLength(output_tensors); ++i) {
            auto name = static_cast<jstring>(env->GetObjectArrayElement(output_tensors, i));
            require(Snpe_StringList_Append(wanted.value, string(env, name).c_str()) == SNPE_SUCCESS, "Output tensor name");
            env->DeleteLocalRef(name);
        }
        if (env->GetArrayLength(output_tensors))
            require(Snpe_SNPEBuilder_SetOutputTensors(builder.value, wanted.value) == SNPE_SUCCESS, "Set output tensors");
        auto engine = std::make_unique<Engine>();
        engine->snpe = Snpe_SNPEBuilder_Build(builder.value);
        require(engine->snpe, "Build SNPE");
        Owned<Snpe_StringList_Delete> in{Snpe_SNPE_GetInputTensorNames(engine->snpe)};
        Owned<Snpe_StringList_Delete> out{Snpe_SNPE_GetOutputTensorNames(engine->snpe)};
        require(in.value && out.value, "Tensor names");
        engine->inputs = names(in.value); engine->outputs = names(out.value);
        for (const auto& name : engine->inputs) {
            Owned<Snpe_TensorShape_Delete> shape{Snpe_SNPE_GetInputDimensions(engine->snpe, name.c_str())};
            require(shape.value, "Input shape");
            engine->tensors.push_back(Snpe_Util_CreateITensor(shape.value));
            require(engine->tensors.back(), "Create input tensor");
        }
        return reinterpret_cast<jlong>(engine.release());
    } catch (const std::exception& error) { fail(env, error); return 0; }
}

extern "C" JNIEXPORT jobjectArray JNICALL JFN(engineNames)(JNIEnv* env, jobject, jlong handle, jboolean outputs) {
    auto* engine = reinterpret_cast<Engine*>(handle);
    const auto& list = outputs ? engine->outputs : engine->inputs;
    auto result = env->NewObjectArray(list.size(), env->FindClass("java/lang/String"), nullptr);
    for (size_t i = 0; i < list.size(); ++i) env->SetObjectArrayElement(result, i, env->NewStringUTF(list[i].c_str()));
    return result;
}

// inputs in engineNames(false) order; returns outputs in engineNames(true) order.
extern "C" JNIEXPORT jobjectArray JNICALL JFN(engineRun)(JNIEnv* env, jobject, jlong handle, jobjectArray inputs) {
    try {
        auto* engine = reinterpret_cast<Engine*>(handle);
        require(env->GetArrayLength(inputs) == jsize(engine->inputs.size()), "Input count");
        Owned<Snpe_TensorMap_Delete> in{Snpe_TensorMap_Create()}, out{Snpe_TensorMap_Create()};
        for (size_t i = 0; i < engine->inputs.size(); ++i) {
            auto array = static_cast<jfloatArray>(env->GetObjectArrayElement(inputs, i));
            require(size_t(env->GetArrayLength(array)) == Snpe_ITensor_GetSize(engine->tensors[i]), "Input size");
            env->GetFloatArrayRegion(array, 0, env->GetArrayLength(array), static_cast<float*>(Snpe_ITensor_GetData(engine->tensors[i])));
            env->DeleteLocalRef(array);
            Snpe_TensorMap_Add(in.value, engine->inputs[i].c_str(), engine->tensors[i]);
        }
        require(Snpe_SNPE_ExecuteITensors(engine->snpe, in.value, out.value) == SNPE_SUCCESS, "Execute");
        auto result = env->NewObjectArray(engine->outputs.size(), env->FindClass("[F"), nullptr);
        for (size_t i = 0; i < engine->outputs.size(); ++i) {
            auto tensor = Snpe_TensorMap_GetTensor_Ref(out.value, engine->outputs[i].c_str());
            require(tensor, "Output tensor");
            const auto n = Snpe_ITensor_GetSize(tensor);
            auto array = env->NewFloatArray(n);
            env->SetFloatArrayRegion(array, 0, n, static_cast<float*>(Snpe_ITensor_GetData(tensor)));
            env->SetObjectArrayElement(result, i, array);
            env->DeleteLocalRef(array);
        }
        return result;
    } catch (const std::exception& error) { fail(env, error); return nullptr; }
}

extern "C" JNIEXPORT void JNICALL JFN(engineDestroy)(JNIEnv*, jobject, jlong handle) {
    delete reinterpret_cast<Engine*>(handle);
}

// simmax / simtop: handles of HTP engines for the other-object max and the own top-6, or 0 for the CPU (the HTP graphs are
// 384-wide). dim: feature width. Returns the memory [objects * parts * dim] followed by 3 timings in ms (other-object max,
// own top-6, selection and k-means). Optional outputs hold retained row indices padded with -1 and CPU comparison counts.
// Cache input describes the unchanged teaching prefix. Next-cache output captures the actual traversal and covered
// prefixes of this build, so an append can provide the input for another append without modifying the old cache.
extern "C" JNIEXPORT jfloatArray JNICALL JFN(buildMemory)(JNIEnv* env, jobject, jfloatArray patches, jintArray counts, jfloat keep,
                                                          jint parts, jlong simmax, jlong simtop, jint dim, jboolean cutoff,
                                                          jintArray retained, jlongArray diagnostics, jfloatArray cache_scores,
                                                          jintArray cache_witness, jintArray cache_order, jint cached_objects,
                                                          jfloatArray next_scores, jintArray next_witness, jintArray next_order) {
    try {
        require(dim > 0 && dim % 16 == 0, "Feature width must be a positive multiple of 16");
        require((simmax == 0 && simtop == 0) || dim == 384, "HTP memory graphs are 384-wide");
        require(!cutoff || (dim == 768 && simmax == 0 && simtop == 0), "Cutoff requires the B CPU memory build");
        require(!diagnostics || (env->GetArrayLength(diagnostics) >= 3 && simmax == 0 && simtop == 0), "CPU comparison diagnostics");
        const std::lock_guard<std::mutex> lock(memory_build_lock);
        D = dim;
        std::vector<float> p(env->GetArrayLength(patches));
        env->GetFloatArrayRegion(patches, 0, p.size(), p.data());
        std::vector<int> c(env->GetArrayLength(counts));
        env->GetIntArrayRegion(counts, 0, c.size(), reinterpret_cast<jint*>(c.data()));
        const int rows = std::accumulate(c.begin(), c.end(), 0);
        require(size_t(rows) * D == p.size(), "Patch count");
        require(!retained || env->GetArrayLength(retained) == rows, "Retained index array must match patch rows");
        // Features are unit-normalized. The looser bound admits any finite values that cannot overflow the dot or mean.
        const bool use_cutoff = cutoff && c.size() >= 2 && keep > 0 && keep <= 1 &&
            std::all_of(c.begin(), c.end(), [](int n) { return n >= 6; }) &&
            std::all_of(p.begin(), p.end(), [](float v) { return std::abs(v) <= 1e15f; });
        require(cached_objects == 0 ? !cache_scores && !cache_witness && !cache_order :
                cache_scores && cache_witness && cache_order && cached_objects >= 2 && cached_objects < int(c.size()),
                "Teaching cache input must describe the objects before the append");
        require(next_scores ? next_witness && next_order : !next_witness && !next_order,
                "All next teaching cache arrays are required");
        require((!cache_scores && !next_scores) || use_cutoff, "Teaching cache requires the B CPU cutoff build");
        TeachingCache cache;
        if (cache_scores) {
            cache.objects = cached_objects;
            const int cached_rows = std::accumulate(c.begin(), c.begin() + cached_objects, 0);
            const int slots = 2 * cached_rows;
            const int order_size = negative_order_size(c, cached_objects, cached_rows);
            require(env->GetArrayLength(cache_scores) == slots && env->GetArrayLength(cache_witness) == slots &&
                    env->GetArrayLength(cache_order) == order_size, "Teaching cache input size");
            cache.scores.resize(slots);
            cache.coverage.resize(slots);
            cache.order.resize(order_size);
            env->GetFloatArrayRegion(cache_scores, 0, slots, cache.scores.data());
            env->GetIntArrayRegion(cache_witness, 0, slots, cache.coverage.data());
            env->GetIntArrayRegion(cache_order, 0, order_size, cache.order.data());
        }
        TeachingCache next;
        if (next_scores) {
            const int slots = 2 * rows;
            const int order_size = negative_order_size(c, int(c.size()), rows);
            require(env->GetArrayLength(next_scores) == slots && env->GetArrayLength(next_witness) == slots &&
                    env->GetArrayLength(next_order) == order_size, "Next teaching cache output size");
            next.scores.resize(slots);
            next.coverage.resize(slots);
            next.order.resize(order_size);
            for (int i = 0; i < slots; i += 2) next.coverage[i] = -1;
        }
        double ms[3] = {0, 0, 0};
        auto* engine = reinterpret_cast<Engine*>(simmax);
        OtherMax om = engine ? OtherMax([engine](const float* P, int n, const float* all, const std::vector<int>& o, std::vector<float>& s) {
                                   htp_other_max(engine, P, n, all, o, s); })
                             : OtherMax(cpu_other_max);
        std::vector<jint> indices;
        MemoryDiagnostics counts_out;
        auto mem = build_memory(p, c, keep, parts, om, ms, reinterpret_cast<Engine*>(simtop), use_cutoff,
                                retained ? &indices : nullptr, counts_out, cache, next_scores ? &next : nullptr);
        if (next_scores) {
            const int slots = static_cast<int>(next.scores.size());
            env->SetFloatArrayRegion(next_scores, 0, slots, next.scores.data());
            env->SetIntArrayRegion(next_witness, 0, slots, next.coverage.data());
            env->SetIntArrayRegion(next_order, 0, next.order.size(), next.order.data());
        }
        if (retained) {
            indices.resize(rows, -1);
            env->SetIntArrayRegion(retained, 0, rows, indices.data());
        }
        if (diagnostics) {
            const jlong values[] = {counts_out.negative_dots, counts_out.positive_dots, counts_out.pruned_rows,
                                    counts_out.reused_witness_rows, counts_out.ineligible_witness_rows,
                                    counts_out.covered_negative_dots};
            env->SetLongArrayRegion(diagnostics, 0, env->GetArrayLength(diagnostics) >= 6 ? 6 : 3, values);
        }
        for (double v : ms) mem.push_back(float(v));
        auto result = env->NewFloatArray(mem.size());
        env->SetFloatArrayRegion(result, 0, mem.size(), mem.data());
        return result;
    } catch (const std::exception& error) { fail(env, error); return nullptr; }
}

extern "C" JNIEXPORT jfloatArray JNICALL JFN(roiAlign)(JNIEnv* env, jobject, jfloatArray image, jint size,
                                                       jfloat x0, jfloat y0, jfloat x1, jfloat y1, jint out) {
    std::vector<float> img(env->GetArrayLength(image));
    env->GetFloatArrayRegion(image, 0, img.size(), img.data());
    auto res = roi_align(img.data(), 3, size, size, x0, y0, x1, y1, out);
    auto result = env->NewFloatArray(res.size());
    env->SetFloatArrayRegion(result, 0, res.size(), res.data());
    return result;
}
