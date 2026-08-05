#include <jni.h>
#include <android/log.h>
#include <arm_neon.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace {
constexpr int kInputWidth = 640;
constexpr int kInputHeight = 480;
constexpr int kOutputWidth = 686;
constexpr int kOutputHeight = 518;
constexpr int kInputBytes = kInputWidth * kInputHeight * 3;
constexpr int kOutputElements = kOutputWidth * kOutputHeight * 3;
constexpr int kAlignedDepthWidth = 640;
constexpr int kAlignedDepthHeight = 480;
thread_local std::vector<float> decoded_depth_workspace(
    static_cast<size_t>(kOutputWidth) * kOutputHeight);

struct DepthAlignMap {
    std::vector<int> x0 = std::vector<int>(kAlignedDepthWidth);
    std::vector<int> x1 = std::vector<int>(kAlignedDepthWidth);
    std::vector<double> fx = std::vector<double>(kAlignedDepthWidth);
    std::vector<int> y0 = std::vector<int>(kAlignedDepthHeight);
    std::vector<int> y1 = std::vector<int>(kAlignedDepthHeight);
    std::vector<double> fy = std::vector<double>(kAlignedDepthHeight);

    DepthAlignMap() {
        for (int column = 0; column < kAlignedDepthWidth; ++column) {
            const double source = static_cast<double>(column) * (kOutputWidth - 1) /
                                  (kAlignedDepthWidth - 1);
            x0[column] = static_cast<int>(source);
            x1[column] = std::min(x0[column] + 1, kOutputWidth - 1);
            fx[column] = source - x0[column];
        }
        for (int row = 0; row < kAlignedDepthHeight; ++row) {
            const double source = static_cast<double>(row) * (kOutputHeight - 1) /
                                  (kAlignedDepthHeight - 1);
            y0[row] = static_cast<int>(source);
            y1[row] = std::min(y0[row] + 1, kOutputHeight - 1);
            fy[row] = source - y0[row];
        }
    }
};

const DepthAlignMap depth_align_map;

void interpolate_cubic(float x, float* coefficients) {
    constexpr float a = -0.75f;
    coefficients[0] = ((a * (x + 1.0f) - 5.0f * a) * (x + 1.0f) + 8.0f * a) *
        (x + 1.0f) - 4.0f * a;
    coefficients[1] = ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
    coefficients[2] = ((a + 2.0f) * (1.0f - x) - (a + 3.0f)) *
        (1.0f - x) * (1.0f - x) + 1.0f;
    coefficients[3] = 1.0f - coefficients[0] - coefficients[1] - coefficients[2];
}

void build_cubic_table(int input_size, int output_size,
                       std::vector<int>& indices, std::vector<float>& coefficients) {
    indices.resize(output_size * 4);
    coefficients.resize(output_size * 4);
    const double scale = static_cast<double>(input_size) / output_size;
    for (int destination = 0; destination < output_size; ++destination) {
        float coordinate = static_cast<float>((destination + 0.5) * scale - 0.5);
        const int base = static_cast<int>(std::floor(coordinate));
        coordinate -= static_cast<float>(base);
        interpolate_cubic(coordinate, coefficients.data() + destination * 4);
        for (int tap = 0; tap < 4; ++tap) {
            indices[destination * 4 + tap] =
                std::clamp(base - 1 + tap, 0, input_size - 1);
        }
    }
}

struct Preprocessor {
    std::vector<uint8_t> rgb = std::vector<uint8_t>(kInputBytes);
    cv::Mat rgb_u8{kInputHeight, kInputWidth, CV_8UC3, rgb.data()};
    cv::Mat rgb_f32{kInputHeight, kInputWidth, CV_32FC3};
    cv::Mat resized_f32{kOutputHeight, kOutputWidth, CV_32FC3};
    std::vector<int> x_indices;
    std::vector<float> x_coefficients;
    std::vector<int> y_indices;
    std::vector<float> y_coefficients;
    std::vector<double> horizontal =
        std::vector<double>(kInputHeight * kOutputWidth * 3);

    Preprocessor() {
        cv::setNumThreads(4);
        build_cubic_table(kInputWidth, kOutputWidth, x_indices, x_coefficients);
        build_cubic_table(kInputHeight, kOutputHeight, y_indices, y_coefficients);
    }
};

void pack_f32(const cv::Mat& input, float* output) {
    constexpr float means[] = {0.485f, 0.456f, 0.406f};
    constexpr float inverse_std[] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};
    constexpr int plane = kOutputWidth * kOutputHeight;
    cv::parallel_for_(cv::Range(0, kOutputHeight), [&](const cv::Range& range) {
        const float32x4_t mean_r = vdupq_n_f32(means[0]);
        const float32x4_t mean_g = vdupq_n_f32(means[1]);
        const float32x4_t mean_b = vdupq_n_f32(means[2]);
        const float32x4_t inv_r = vdupq_n_f32(inverse_std[0]);
        const float32x4_t inv_g = vdupq_n_f32(inverse_std[1]);
        const float32x4_t inv_b = vdupq_n_f32(inverse_std[2]);
        for (int row = range.start; row < range.end; ++row) {
            const float* source = input.ptr<float>(row);
            float* red = output + row * kOutputWidth;
            float* green = output + plane + row * kOutputWidth;
            float* blue = output + 2 * plane + row * kOutputWidth;
            int column = 0;
            for (; column <= kOutputWidth - 4; column += 4) {
                const float32x4x3_t rgb = vld3q_f32(source + column * 3);
                vst1q_f32(red + column, vmulq_f32(vsubq_f32(rgb.val[0], mean_r), inv_r));
                vst1q_f32(green + column, vmulq_f32(vsubq_f32(rgb.val[1], mean_g), inv_g));
                vst1q_f32(blue + column, vmulq_f32(vsubq_f32(rgb.val[2], mean_b), inv_b));
            }
            for (; column < kOutputWidth; ++column) {
                red[column] = (source[column * 3] - means[0]) * inverse_std[0];
                green[column] = (source[column * 3 + 1] - means[1]) * inverse_std[1];
                blue[column] = (source[column * 3 + 2] - means[2]) * inverse_std[2];
            }
        }
    });
}

void pack_f16(const cv::Mat& input, __fp16* output) {
    constexpr float means[] = {0.485f, 0.456f, 0.406f};
    constexpr float inverse_std[] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};
    constexpr int plane = kOutputWidth * kOutputHeight;
    cv::parallel_for_(cv::Range(0, kOutputHeight), [&](const cv::Range& range) {
        const float32x4_t mean_r = vdupq_n_f32(means[0]);
        const float32x4_t mean_g = vdupq_n_f32(means[1]);
        const float32x4_t mean_b = vdupq_n_f32(means[2]);
        const float32x4_t inv_r = vdupq_n_f32(inverse_std[0]);
        const float32x4_t inv_g = vdupq_n_f32(inverse_std[1]);
        const float32x4_t inv_b = vdupq_n_f32(inverse_std[2]);
        for (int row = range.start; row < range.end; ++row) {
            const float* source = input.ptr<float>(row);
            __fp16* red = output + row * kOutputWidth;
            __fp16* green = output + plane + row * kOutputWidth;
            __fp16* blue = output + 2 * plane + row * kOutputWidth;
            int column = 0;
            for (; column <= kOutputWidth - 4; column += 4) {
                const float32x4x3_t rgb = vld3q_f32(source + column * 3);
                vst1_f16(red + column, vcvt_f16_f32(vmulq_f32(vsubq_f32(rgb.val[0], mean_r), inv_r)));
                vst1_f16(green + column, vcvt_f16_f32(vmulq_f32(vsubq_f32(rgb.val[1], mean_g), inv_g)));
                vst1_f16(blue + column, vcvt_f16_f32(vmulq_f32(vsubq_f32(rgb.val[2], mean_b), inv_b)));
            }
            for (; column < kOutputWidth; ++column) {
                red[column] = static_cast<__fp16>((source[column * 3] - means[0]) * inverse_std[0]);
                green[column] = static_cast<__fp16>((source[column * 3 + 1] - means[1]) * inverse_std[1]);
                blue[column] = static_cast<__fp16>((source[column * 3 + 2] - means[2]) * inverse_std[2]);
            }
        }
    });
}

void canonical_official_f32(const uint8_t* input, Preprocessor& preprocessor, float* output) {
    constexpr double means[] = {0.485, 0.456, 0.406};
    constexpr double stds[] = {0.229, 0.224, 0.225};
    constexpr int plane = kOutputWidth * kOutputHeight;
    cv::parallel_for_(cv::Range(0, kInputHeight), [&](const cv::Range& range) {
        for (int row = range.start; row < range.end; ++row) {
            for (int column = 0; column < kOutputWidth; ++column) {
                const int table = column * 4;
                for (int channel = 0; channel < 3; ++channel) {
                    const auto sample = [&](int tap) {
                        const int source_column = preprocessor.x_indices[table + tap];
                        return static_cast<double>(input[(row * kInputWidth + source_column) * 3 + channel]) /
                            255.0;
                    };
                    double value = sample(0) * static_cast<double>(preprocessor.x_coefficients[table]);
                    value = value + sample(1) * static_cast<double>(preprocessor.x_coefficients[table + 1]);
                    value = value + sample(2) * static_cast<double>(preprocessor.x_coefficients[table + 2]);
                    value = value + sample(3) * static_cast<double>(preprocessor.x_coefficients[table + 3]);
                    preprocessor.horizontal[(row * kOutputWidth + column) * 3 + channel] = value;
                }
            }
        }
    });
    cv::parallel_for_(cv::Range(0, kOutputHeight), [&](const cv::Range& range) {
        for (int row = range.start; row < range.end; ++row) {
            float* red = output + row * kOutputWidth;
            float* green = output + plane + row * kOutputWidth;
            float* blue = output + 2 * plane + row * kOutputWidth;
            const int table = row * 4;
            for (int column = 0; column < kOutputWidth; ++column) {
                for (int channel = 0; channel < 3; ++channel) {
                    const auto sample = [&](int tap) {
                        const int source_row = preprocessor.y_indices[table + tap];
                        return preprocessor.horizontal[
                            (source_row * kOutputWidth + column) * 3 + channel];
                    };
                    double value = sample(0) * static_cast<double>(preprocessor.y_coefficients[table]);
                    value = value + sample(1) * static_cast<double>(preprocessor.y_coefficients[table + 1]);
                    value = value + sample(2) * static_cast<double>(preprocessor.y_coefficients[table + 2]);
                    value = value + sample(3) * static_cast<double>(preprocessor.y_coefficients[table + 3]);
                    const float normalized = static_cast<float>((value - means[channel]) / stds[channel]);
                    output[channel * plane + row * kOutputWidth + column] = normalized;
                }
            }
        }
    });
}

uint16_t float_to_half_rte(float value) {
    uint32_t bits;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t mantissa = bits & 0x7fffffu;

    if (exponent == 0xffu) {
        if (mantissa == 0) return static_cast<uint16_t>(sign | 0x7c00u);
        // Preserve the representable payload and ensure NaN never becomes Inf.
        uint16_t payload = static_cast<uint16_t>(mantissa >> 13);
        if (payload == 0) payload = 1;
        return static_cast<uint16_t>(sign | 0x7c00u | payload);
    }

    // Binary32 subnormals and values with an unbiased exponent below -25
    // cannot round to a non-zero binary16 value.
    if (exponent == 0 || exponent < 102u) return sign;
    const int unbiased = static_cast<int>(exponent) - 127;

    if (unbiased < -14) {
        const uint32_t significand = mantissa | 0x800000u;
        const int shift = -unbiased - 1;  // 14..24 for e=-15..-25.
        uint32_t rounded = significand >> shift;
        const uint32_t remainder_mask = (1u << shift) - 1u;
        const uint32_t remainder = significand & remainder_mask;
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u) != 0)) ++rounded;
        return static_cast<uint16_t>(sign | rounded);
    }

    uint32_t half_exponent = static_cast<uint32_t>(unbiased + 15);
    uint32_t half_mantissa = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_mantissa & 1u) != 0)) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0;
            ++half_exponent;
        }
    }
    if (half_exponent >= 31u) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (half_exponent << 10) | half_mantissa);
}

float half_to_float_exact(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            bits = sign;
        } else {
            int unbiased_exponent = -14;
            while ((mantissa & 0x0400u) == 0u) {
                mantissa <<= 1;
                --unbiased_exponent;
            }
            mantissa &= 0x03ffu;
            bits = sign |
                (static_cast<uint32_t>(unbiased_exponent + 127) << 23) |
                (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
        if (mantissa != 0u) bits |= 0x00400000u;
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float output;
    std::memcpy(&output, &bits, sizeof(output));
    return output;
}
}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_linnan_blindassist_hftf_Dav2NativePreprocessor_nativeCreate(JNIEnv*, jobject) {
    return reinterpret_cast<jlong>(new Preprocessor());
}

extern "C" JNIEXPORT void JNICALL
Java_com_linnan_blindassist_hftf_Dav2NativePreprocessor_nativeRun(
        JNIEnv* env, jobject, jlong handle, jbyteArray input, jobject output, jboolean fp16) {
    auto* preprocessor = reinterpret_cast<Preprocessor*>(handle);
    if (preprocessor == nullptr || env->GetArrayLength(input) != kInputBytes) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "invalid native preprocessor input");
        return;
    }
    void* output_address = env->GetDirectBufferAddress(output);
    const jlong capacity = env->GetDirectBufferCapacity(output);
    const jlong required = static_cast<jlong>(kOutputElements) * (fp16 ? 2 : 4);
    if (output_address == nullptr || capacity < required) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "invalid native direct output buffer");
        return;
    }
    env->GetByteArrayRegion(input, 0, kInputBytes, reinterpret_cast<jbyte*>(preprocessor->rgb.data()));
    preprocessor->rgb_u8.convertTo(preprocessor->rgb_f32, CV_32FC3, 1.0 / 255.0);
    cv::resize(
        preprocessor->rgb_f32,
        preprocessor->resized_f32,
        cv::Size(kOutputWidth, kOutputHeight),
        0.0,
        0.0,
        cv::INTER_CUBIC
    );
    if (fp16) {
        pack_f16(preprocessor->resized_f32, reinterpret_cast<__fp16*>(output_address));
    } else {
        pack_f32(preprocessor->resized_f32, reinterpret_cast<float*>(output_address));
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_linnan_blindassist_hftf_Dav2NativePreprocessor_nativeRunOfficial(
        JNIEnv* env, jobject, jlong handle, jbyteArray input, jobject output) {
    auto* preprocessor = reinterpret_cast<Preprocessor*>(handle);
    if (preprocessor == nullptr || env->GetArrayLength(input) != kInputBytes) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                      "invalid official-compatible preprocessor input");
        return;
    }
    void* output_address = env->GetDirectBufferAddress(output);
    const jlong capacity = env->GetDirectBufferCapacity(output);
    const jlong required = static_cast<jlong>(kOutputElements) * 4;
    if (output_address == nullptr || capacity < required) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                      "invalid official-compatible output buffer");
        return;
    }
    env->GetByteArrayRegion(input, 0, kInputBytes,
                            reinterpret_cast<jbyte*>(preprocessor->rgb.data()));
    canonical_official_f32(preprocessor->rgb.data(), *preprocessor,
                           reinterpret_cast<float*>(output_address));
}

extern "C" JNIEXPORT void JNICALL
Java_com_linnan_blindassist_hftf_Dav2NativePreprocessor_nativeRunOfficialDirect(
        JNIEnv* env, jobject, jlong handle, jobject input, jobject output) {
    auto* preprocessor = reinterpret_cast<Preprocessor*>(handle);
    const auto* input_address = static_cast<const uint8_t*>(env->GetDirectBufferAddress(input));
    void* output_address = env->GetDirectBufferAddress(output);
    const jlong input_capacity = env->GetDirectBufferCapacity(input);
    const jlong output_capacity = env->GetDirectBufferCapacity(output);
    const jlong required_output = static_cast<jlong>(kOutputElements) * 4;
    if (preprocessor == nullptr || input_address == nullptr || output_address == nullptr ||
        input_capacity < kInputBytes || output_capacity < required_output) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                      "invalid direct official-compatible buffers");
        return;
    }
    canonical_official_f32(input_address, *preprocessor,
                           reinterpret_cast<float*>(output_address));
}

extern "C" JNIEXPORT void JNICALL
Java_com_linnan_blindassist_hftf_Dav2NativePreprocessor_nativeConvertFp32ToFp16(
        JNIEnv* env, jobject, jobject input, jobject output, jint elements) {
    const auto* input_address = static_cast<const float*>(env->GetDirectBufferAddress(input));
    auto* output_address = static_cast<uint16_t*>(env->GetDirectBufferAddress(output));
    const jlong input_capacity = env->GetDirectBufferCapacity(input);
    const jlong output_capacity = env->GetDirectBufferCapacity(output);
    if (elements < 0 || input_address == nullptr || output_address == nullptr ||
        input_capacity < static_cast<jlong>(elements) * 4 ||
        output_capacity < static_cast<jlong>(elements) * 2) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                      "invalid strict FP32-to-FP16 buffers");
        return;
    }
    for (jint index = 0; index < elements; ++index) {
        output_address[index] = float_to_half_rte(input_address[index]);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_linnan_blindassist_hftf_Dav2NativePreprocessor_nativeDecodeFp16ToFp32(
        JNIEnv* env, jobject, jobject input, jfloatArray output, jint elements) {
    const auto* input_address = static_cast<const uint16_t*>(env->GetDirectBufferAddress(input));
    const jlong input_capacity = env->GetDirectBufferCapacity(input);
    if (elements < 0 || input_address == nullptr ||
        input_capacity < static_cast<jlong>(elements) * 2 ||
        env->GetArrayLength(output) < elements) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                      "invalid strict FP16-to-FP32 buffers");
        return;
    }
    auto* output_address = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(output, nullptr));
    if (output_address == nullptr) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                      "unable to pin FP32 output array");
        return;
    }
    for (jint index = 0; index < elements; ++index) {
        output_address[index] = half_to_float_exact(input_address[index]);
    }
    env->ReleasePrimitiveArrayCritical(output, output_address, 0);
}

extern "C" JNIEXPORT void JNICALL
Java_com_linnan_blindassist_hftf_Dav2NativePreprocessor_nativeDecodeResizeFp16AlignCorners(
        JNIEnv* env, jobject, jobject input, jobject output) {
    const auto* input_address = static_cast<const uint16_t*>(env->GetDirectBufferAddress(input));
    auto* output_address = static_cast<float*>(env->GetDirectBufferAddress(output));
    const jlong input_capacity = env->GetDirectBufferCapacity(input);
    const jlong output_capacity = env->GetDirectBufferCapacity(output);
    if (input_address == nullptr || output_address == nullptr ||
        input_capacity < static_cast<jlong>(kOutputWidth) * kOutputHeight * 2 ||
        output_capacity < static_cast<jlong>(kAlignedDepthWidth) * kAlignedDepthHeight * 4) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                      "invalid direct FP16 depth or aligned-depth buffer");
        return;
    }
    for (int index = 0; index < kOutputWidth * kOutputHeight; ++index) {
        decoded_depth_workspace[index] = half_to_float_exact(input_address[index]);
    }
    const float* decoded = decoded_depth_workspace.data();
    for (int row = 0; row < kAlignedDepthHeight; ++row) {
        const int y0 = depth_align_map.y0[row];
        const int y1 = depth_align_map.y1[row];
        const double fy = depth_align_map.fy[row];
        for (int column = 0; column < kAlignedDepthWidth; ++column) {
            const int x0 = depth_align_map.x0[column];
            const int x1 = depth_align_map.x1[column];
            const double fx = depth_align_map.fx[column];
            const double top = decoded[y0 * kOutputWidth + x0] * (1.0 - fx) +
                               decoded[y0 * kOutputWidth + x1] * fx;
            const double bottom = decoded[y1 * kOutputWidth + x0] * (1.0 - fx) +
                                  decoded[y1 * kOutputWidth + x1] * fx;
            output_address[row * kAlignedDepthWidth + column] =
                static_cast<float>(top * (1.0 - fy) + bottom * fy);
        }
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_linnan_blindassist_hftf_Dav2NativePreprocessor_nativeCopyLastResizedHwcFp32(
        JNIEnv* env, jobject, jlong handle, jobject output) {
    auto* preprocessor = reinterpret_cast<Preprocessor*>(handle);
    void* output_address = env->GetDirectBufferAddress(output);
    const jlong output_capacity = env->GetDirectBufferCapacity(output);
    const jlong required = static_cast<jlong>(kOutputElements) * 4;
    if (preprocessor == nullptr || output_address == nullptr || output_capacity < required) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                      "invalid resized FP32 diagnostic buffer");
        return;
    }
    std::memcpy(output_address, preprocessor->resized_f32.ptr<float>(), required);
}

extern "C" JNIEXPORT void JNICALL
Java_com_linnan_blindassist_hftf_Dav2NativePreprocessor_nativeDestroy(
        JNIEnv*, jobject, jlong handle) {
    delete reinterpret_cast<Preprocessor*>(handle);
}
