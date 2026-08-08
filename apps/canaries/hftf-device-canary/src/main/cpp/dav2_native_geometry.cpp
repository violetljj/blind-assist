#include <jni.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
constexpr double kLowerRoiStartFraction = 0.55;
constexpr int kStride = 4;
constexpr int64_t kRansacSeed = 1729;
constexpr int kRansacIterations = 240;
constexpr int kMaximumCandidates = 5000;
constexpr int kMinimumCandidates = 100;
constexpr int kMinimumInliers = 80;
constexpr double kMinimumInlierFraction = 0.08;
constexpr double kMinimumAbsNormalY = 0.55;
constexpr double kMaximumNormalizedPlaneResidual = 0.035;
constexpr double kMinimumScale = 0.25;
constexpr double kMaximumScale = 4.0;
constexpr int kResultSize = 17;

enum Status {
    kValid = 0,
    kInvalidInput = 1,
    kInsufficientGroundCandidates = 2,
    kDegenerateRelativeDepth = 3,
    kNoGroundConsensus = 4,
    kDegenerateRelativeHeight = 5,
    kGroundOrientationRejected = 6,
    kGroundSupportRejected = 7,
    kGroundResidualRejected = 8,
    kScaleOutOfRange = 9,
    kInsufficientValidDepth = 10,
};

struct JavaRandom {
    explicit JavaRandom(int64_t seed)
        : state((static_cast<uint64_t>(seed) ^ kMultiplier) & kMask) {}

    int next_int(int bound) {
        if ((bound & -bound) == bound) {
            return static_cast<int>((static_cast<int64_t>(bound) * next(31)) >> 31);
        }
        int bits;
        int value;
        do {
            bits = next(31);
            value = bits % bound;
        } while (static_cast<int32_t>(bits - value + (bound - 1)) < 0);
        return value;
    }

private:
    int next(int bits) {
        state = (state * kMultiplier + kAddend) & kMask;
        return static_cast<int>(state >> (48 - bits));
    }
    static constexpr uint64_t kMultiplier = 0x5DEECE66DULL;
    static constexpr uint64_t kAddend = 0xBULL;
    static constexpr uint64_t kMask = (1ULL << 48) - 1;
    uint64_t state;
};

struct Workspace {
    std::vector<double> candidate_x;
    std::vector<double> candidate_y;
    std::vector<double> candidate_z;
    std::array<double, kMaximumCandidates> sampled_x{};
    std::array<double, kMaximumCandidates> sampled_y{};
    std::array<double, kMaximumCandidates> sampled_z{};
    std::array<int, kMaximumCandidates> first_indices{};
    std::array<int, kMaximumCandidates> second_indices{};
    std::array<double, kMaximumCandidates> residuals{};
    std::vector<double> finite_depth;
};

thread_local Workspace workspace;

double select_value(double* values, int count, int target) {
    std::nth_element(values, values + target, values + count);
    return values[target];
}

double median(double* values, int count) {
    const int middle = count / 2;
    if ((count & 1) != 0) return select_value(values, count, middle);
    const double upper = select_value(values, count, middle);
    const double lower = select_value(values, count, middle - 1);
    return 0.5 * (lower + upper);
}

double quantile(double* values, int count, double fraction) {
    const double position = fraction * (count - 1);
    const int lower_index = static_cast<int>(position);
    const int upper_index = std::min(lower_index + 1, count - 1);
    const double lower = select_value(values, count, lower_index);
    const double upper = upper_index == lower_index ? lower : select_value(values, count, upper_index);
    const double weight = position - lower_index;
    return lower * (1.0 - weight) + upper * weight;
}

std::array<double, 3> smallest_eigenvector(const std::array<std::array<double, 3>, 3>& input) {
    auto matrix = input;
    std::array<std::array<double, 3>, 3> vectors{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
    for (int iteration = 0; iteration < 32; ++iteration) {
        int p = 0;
        int q = 1;
        if (std::abs(matrix[0][2]) > std::abs(matrix[p][q])) { p = 0; q = 2; }
        if (std::abs(matrix[1][2]) > std::abs(matrix[p][q])) { p = 1; q = 2; }
        if (std::abs(matrix[p][q]) < 1e-12) continue;
        const double angle = 0.5 * std::atan2(2.0 * matrix[p][q], matrix[q][q] - matrix[p][p]);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        for (int index = 0; index < 3; ++index) {
            const double mip = matrix[index][p];
            const double miq = matrix[index][q];
            matrix[index][p] = cosine * mip - sine * miq;
            matrix[index][q] = sine * mip + cosine * miq;
        }
        for (int index = 0; index < 3; ++index) {
            const double mpi = matrix[p][index];
            const double mqi = matrix[q][index];
            matrix[p][index] = cosine * mpi - sine * mqi;
            matrix[q][index] = sine * mpi + cosine * mqi;
        }
        for (int index = 0; index < 3; ++index) {
            const double vip = vectors[index][p];
            const double viq = vectors[index][q];
            vectors[index][p] = cosine * vip - sine * viq;
            vectors[index][q] = sine * vip + cosine * viq;
        }
    }
    int eigen_index = 0;
    if (matrix[1][1] < matrix[eigen_index][eigen_index]) eigen_index = 1;
    if (matrix[2][2] < matrix[eigen_index][eigen_index]) eigen_index = 2;
    std::array<double, 3> result{vectors[0][eigen_index], vectors[1][eigen_index], vectors[2][eigen_index]};
    const double norm = std::sqrt(result[0] * result[0] + result[1] * result[1] + result[2] * result[2]);
    for (double& value : result) value /= norm;
    return result;
}

std::array<double, kResultSize> evaluate(
        const float* depth, int depth_size, int width, int height,
        double fx, double fy, double cx, double cy, double camera_height) {
    std::array<double, kResultSize> result{};
    auto reject = [&](Status status) { result[0] = status; return result; };
    if (depth_size != width * height || width <= 0 || height <= 0 || fx <= 0.0 || fy <= 0.0) {
        return reject(kInvalidInput);
    }

    const int start_row = static_cast<int>(std::ceil(kLowerRoiStartFraction * height));
    const int rows = (height - start_row + kStride - 1) / kStride;
    const int columns = (width + kStride - 1) / kStride;
    const int capacity = rows * columns;
    workspace.candidate_x.resize(capacity);
    workspace.candidate_y.resize(capacity);
    workspace.candidate_z.resize(capacity);
    int candidate_count = 0;
    for (int row = start_row; row < height; row += kStride) {
        for (int column = 0; column < width; column += kStride) {
            const double z = depth[row * width + column];
            if (std::isfinite(z) && z > 0.0) {
                workspace.candidate_x[candidate_count] = (column - cx) * z / fx;
                workspace.candidate_y[candidate_count] = (row - cy) * z / fy;
                workspace.candidate_z[candidate_count] = z;
                ++candidate_count;
            }
        }
    }
    if (candidate_count < kMinimumCandidates) return reject(kInsufficientGroundCandidates);

    int point_count = candidate_count;
    const double* px = workspace.candidate_x.data();
    const double* py = workspace.candidate_y.data();
    const double* pz = workspace.candidate_z.data();
    if (candidate_count > kMaximumCandidates) {
        point_count = kMaximumCandidates;
        for (int index = 0; index < kMaximumCandidates; ++index) {
            const int selected = static_cast<int>(static_cast<int64_t>(index) * (candidate_count - 1) /
                                                  (kMaximumCandidates - 1));
            workspace.sampled_x[index] = px[selected];
            workspace.sampled_y[index] = py[selected];
            workspace.sampled_z[index] = pz[selected];
        }
        px = workspace.sampled_x.data();
        py = workspace.sampled_y.data();
        pz = workspace.sampled_z.data();
    }
    for (int index = 0; index < point_count; ++index) {
        workspace.residuals[index] = std::sqrt(px[index] * px[index] + py[index] * py[index] + pz[index] * pz[index]);
    }
    const double characteristic = median(workspace.residuals.data(), point_count);
    if (!std::isfinite(characteristic) || characteristic <= 0.0) return reject(kDegenerateRelativeDepth);
    const double minimum_height = std::max(std::numeric_limits<double>::denorm_min(), characteristic * 1e-6);
    const double minimum_cross_norm = std::max(std::numeric_limits<double>::denorm_min(), characteristic * characteristic * 1e-12);

    JavaRandom random(kRansacSeed);
    int* current_indices = workspace.first_indices.data();
    int* best_indices = workspace.second_indices.data();
    int best_count = -1;
    double best_residual = std::numeric_limits<double>::infinity();
    for (int iteration = 0; iteration < kRansacIterations; ++iteration) {
        const int a = random.next_int(point_count);
        int b = random.next_int(point_count);
        while (b == a) b = random.next_int(point_count);
        int c = random.next_int(point_count);
        while (c == a || c == b) c = random.next_int(point_count);
        const double abx = px[b] - px[a], aby = py[b] - py[a], abz = pz[b] - pz[a];
        const double acx = px[c] - px[a], acy = py[c] - py[a], acz = pz[c] - pz[a];
        double nx = aby * acz - abz * acy;
        double ny = abz * acx - abx * acz;
        double nz = abx * acy - aby * acx;
        const double norm = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (!std::isfinite(norm) || norm <= minimum_cross_norm) continue;
        nx /= norm; ny /= norm; nz /= norm;
        if (std::abs(ny) < kMinimumAbsNormalY) continue;
        double offset = -(nx * px[a] + ny * py[a] + nz * pz[a]);
        if (offset < 0.0) { nx = -nx; ny = -ny; nz = -nz; offset = -offset; }
        if (!std::isfinite(offset) || offset <= minimum_height) continue;
        int count = 0;
        for (int index = 0; index < point_count; ++index) {
            const double residual = std::abs(nx * px[index] + ny * py[index] + nz * pz[index] + offset) / offset;
            if (residual <= kMaximumNormalizedPlaneResidual) {
                current_indices[count] = index;
                workspace.residuals[count] = residual;
                ++count;
            }
        }
        const double residual = count == 0 ? std::numeric_limits<double>::infinity()
                                           : median(workspace.residuals.data(), count);
        if (count > best_count || (count == best_count && residual < best_residual)) {
            best_count = count;
            best_residual = residual;
            std::swap(current_indices, best_indices);
        }
    }
    const int required = std::max(kMinimumInliers, static_cast<int>(std::ceil(kMinimumInlierFraction * point_count)));
    if (best_count < required) return reject(kNoGroundConsensus);

    double center_x = 0.0, center_y = 0.0, center_z = 0.0;
    for (int position = 0; position < best_count; ++position) {
        const int index = best_indices[position];
        center_x += px[index]; center_y += py[index]; center_z += pz[index];
    }
    center_x /= best_count; center_y /= best_count; center_z /= best_count;
    std::array<std::array<double, 3>, 3> covariance{};
    for (int position = 0; position < best_count; ++position) {
        const int index = best_indices[position];
        const double dx = px[index] - center_x, dy = py[index] - center_y, dz = pz[index] - center_z;
        covariance[0][0] += dx * dx; covariance[0][1] += dx * dy; covariance[0][2] += dx * dz;
        covariance[1][0] += dy * dx; covariance[1][1] += dy * dy; covariance[1][2] += dy * dz;
        covariance[2][0] += dz * dx; covariance[2][1] += dz * dy; covariance[2][2] += dz * dz;
    }
    auto normal = smallest_eigenvector(covariance);
    double offset = -(normal[0] * center_x + normal[1] * center_y + normal[2] * center_z);
    if (offset < 0.0) { for (double& value : normal) value = -value; offset = -offset; }
    if (!std::isfinite(offset) || offset <= minimum_height) return reject(kDegenerateRelativeHeight);
    if (std::abs(normal[1]) < kMinimumAbsNormalY) return reject(kGroundOrientationRejected);
    int accepted_count = 0;
    for (int index = 0; index < point_count; ++index) {
        const double residual = std::abs(normal[0] * px[index] + normal[1] * py[index] + normal[2] * pz[index] + offset) / offset;
        if (residual <= kMaximumNormalizedPlaneResidual) workspace.residuals[accepted_count++] = residual;
    }
    const double inlier_fraction = static_cast<double>(accepted_count) / point_count;
    if (accepted_count < required || inlier_fraction < kMinimumInlierFraction) return reject(kGroundSupportRejected);
    const double median_residual = median(workspace.residuals.data(), accepted_count);
    if (median_residual > kMaximumNormalizedPlaneResidual) return reject(kGroundResidualRejected);
    const double scale = camera_height / offset;
    if (!std::isfinite(scale) || scale < kMinimumScale || scale > kMaximumScale) return reject(kScaleOutOfRange);

    workspace.finite_depth.resize(depth_size);
    int finite_count = 0;
    for (int index = 0; index < depth_size; ++index) {
        const double value = depth[index];
        if (std::isfinite(value) && value > 0.0) workspace.finite_depth[finite_count++] = value;
    }
    if (finite_count < 500) return reject(kInsufficientValidDepth);
    const double q10 = quantile(workspace.finite_depth.data(), finite_count, 0.10);
    const double q50 = quantile(workspace.finite_depth.data(), finite_count, 0.50);
    const double q90 = quantile(workspace.finite_depth.data(), finite_count, 0.90);

    result[0] = kValid;
    result[1] = offset; result[2] = median_residual; result[3] = inlier_fraction;
    result[4] = normal[0]; result[5] = normal[1]; result[6] = normal[2];
    result[7] = std::log(scale); result[8] = std::log(camera_height);
    result[9] = normal[0]; result[10] = normal[1]; result[11] = normal[2];
    result[12] = median_residual;
    result[13] = std::log(q10); result[14] = std::log(q50); result[15] = std::log(q90);
    result[16] = std::log(q90 / q10);
    return result;
}
}  // namespace

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_linnan_blindassist_hftf_Dav2NativeGeometry_nativeEvaluate(
        JNIEnv* env, jobject, jfloatArray depth, jint width, jint height,
        jdouble fx, jdouble fy, jdouble cx, jdouble cy, jdouble camera_height) {
    const jsize depth_size = env->GetArrayLength(depth);
    auto* depth_address = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(depth, nullptr));
    if (depth_address == nullptr) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "unable to pin geometry depth array");
        return nullptr;
    }
    const auto result = evaluate(depth_address, depth_size, width, height, fx, fy, cx, cy, camera_height);
    env->ReleasePrimitiveArrayCritical(depth, depth_address, JNI_ABORT);
    const int result_size = result[0] == kValid ? kResultSize : 1;
    jdoubleArray output = env->NewDoubleArray(result_size);
    if (output != nullptr) env->SetDoubleArrayRegion(output, 0, result_size, result.data());
    return output;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_linnan_blindassist_hftf_Dav2NativeGeometry_nativeEvaluateDirect(
        JNIEnv* env, jobject, jobject depth, jint width, jint height,
        jdouble fx, jdouble fy, jdouble cx, jdouble cy, jdouble camera_height) {
    const auto* depth_address = static_cast<const float*>(env->GetDirectBufferAddress(depth));
    const jlong capacity = env->GetDirectBufferCapacity(depth);
    const int depth_size = width * height;
    if (depth_address == nullptr || width <= 0 || height <= 0 ||
        capacity < static_cast<jlong>(depth_size) * 4) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                      "invalid direct geometry depth buffer");
        return nullptr;
    }
    const auto result = evaluate(depth_address, depth_size, width, height, fx, fy, cx, cy, camera_height);
    const int result_size = result[0] == kValid ? kResultSize : 1;
    jdoubleArray output = env->NewDoubleArray(result_size);
    if (output != nullptr) env->SetDoubleArrayRegion(output, 0, result_size, result.data());
    return output;
}
