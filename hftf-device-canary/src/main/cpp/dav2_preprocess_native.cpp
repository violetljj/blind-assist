#include <jni.h>
#include <android/log.h>
#include <arm_neon.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace {
constexpr int kInputWidth = 640;
constexpr int kInputHeight = 480;
constexpr int kOutputWidth = 686;
constexpr int kOutputHeight = 518;
constexpr int kInputBytes = kInputWidth * kInputHeight * 3;
constexpr int kOutputElements = kOutputWidth * kOutputHeight * 3;

struct Preprocessor {
    std::vector<uint8_t> rgb = std::vector<uint8_t>(kInputBytes);
    cv::Mat rgb_u8{kInputHeight, kInputWidth, CV_8UC3, rgb.data()};
    cv::Mat rgb_f32{kInputHeight, kInputWidth, CV_32FC3};
    cv::Mat resized_f32{kOutputHeight, kOutputWidth, CV_32FC3};

    Preprocessor() { cv::setNumThreads(4); }
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
Java_com_linnan_blindassist_hftf_Dav2NativePreprocessor_nativeDestroy(
        JNIEnv*, jobject, jlong handle) {
    delete reinterpret_cast<Preprocessor*>(handle);
}
