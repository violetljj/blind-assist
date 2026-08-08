#include <jni.h>
#include <android/bitmap.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace {
constexpr int kOutputWidth = 640;
constexpr int kOutputHeight = 480;

class Converter {
public:
    void convert(JNIEnv* env, jbyteArray y_array, jbyteArray u_array, jbyteArray v_array,
                 int width, int height, int rotation_degrees, uint8_t* output) {
        if (width <= 0 || height <= 0 || (width & 1) || (height & 1))
            throw std::runtime_error("YUV dimensions must be positive and even");
        if (rotation_degrees != 0 && rotation_degrees != 90 &&
            rotation_degrees != 180 && rotation_degrees != 270)
            throw std::runtime_error("rotation must be 0, 90, 180, or 270 degrees");
        const int y_bytes = width * height;
        const int chroma_bytes = y_bytes / 4;
        if (env->GetArrayLength(y_array) < y_bytes || env->GetArrayLength(u_array) < chroma_bytes ||
            env->GetArrayLength(v_array) < chroma_bytes || output == nullptr)
            throw std::runtime_error("YUV or RGB array is smaller than the frozen contract");

        ensureSize(width, height);
        jbyte* y = env->GetByteArrayElements(y_array, nullptr);
        jbyte* u = env->GetByteArrayElements(u_array, nullptr);
        jbyte* v = env->GetByteArrayElements(v_array, nullptr);
        if (!y || !u || !v) {
            if (y) env->ReleaseByteArrayElements(y_array, y, JNI_ABORT);
            if (u) env->ReleaseByteArrayElements(u_array, u, JNI_ABORT);
            if (v) env->ReleaseByteArrayElements(v_array, v, JNI_ABORT);
            throw std::runtime_error("unable to pin YUV/RGB arrays");
        }
        std::memcpy(i420_.ptr(0), y, static_cast<size_t>(y_bytes));
        std::memcpy(i420_.ptr(height), u, static_cast<size_t>(chroma_bytes));
        std::memcpy(i420_.ptr(height + height / 4), v, static_cast<size_t>(chroma_bytes));
        env->ReleaseByteArrayElements(y_array, y, JNI_ABORT);
        env->ReleaseByteArrayElements(u_array, u, JNI_ABORT);
        env->ReleaseByteArrayElements(v_array, v, JNI_ABORT);

        cv::cvtColor(i420_, rgb_, cv::COLOR_YUV2RGB_I420);
        cv::Mat oriented;
        if (rotation_degrees == 0) {
            oriented = rgb_;
        } else {
            const int rotate_code = rotation_degrees == 90 ? cv::ROTATE_90_CLOCKWISE :
                (rotation_degrees == 180 ? cv::ROTATE_180 : cv::ROTATE_90_COUNTERCLOCKWISE);
            cv::rotate(rgb_, rotated_, rotate_code);
            oriented = rotated_;
        }
        const double target_aspect = static_cast<double>(kOutputWidth) / kOutputHeight;
        int crop_width = oriented.cols;
        int crop_height = oriented.rows;
        if (static_cast<double>(crop_width) / crop_height > target_aspect)
            crop_width = static_cast<int>(crop_height * target_aspect);
        else
            crop_height = static_cast<int>(crop_width / target_aspect);
        crop_width = std::max(2, crop_width & ~1);
        crop_height = std::max(2, crop_height & ~1);
        const int left = (oriented.cols - crop_width) / 2;
        const int top = (oriented.rows - crop_height) / 2;
        cv::resize(oriented(cv::Rect(left, top, crop_width, crop_height)), output_rgb_,
                   cv::Size(kOutputWidth, kOutputHeight), 0.0, 0.0, cv::INTER_LINEAR);
        std::memcpy(output, output_rgb_.ptr(0), kOutputWidth * kOutputHeight * 3);
    }

private:
    void ensureSize(int width, int height) {
        if (width == width_ && height == height_) return;
        width_ = width;
        height_ = height;
        i420_.create(height + height / 2, width, CV_8UC1);
        rgb_.create(height, width, CV_8UC3);
        rotated_.create(width, height, CV_8UC3);
        output_rgb_.create(kOutputHeight, kOutputWidth, CV_8UC3);
    }
    int width_ = 0;
    int height_ = 0;
    cv::Mat i420_, rgb_, rotated_, output_rgb_;
};

void throwJava(JNIEnv* env, const char* message) {
    jclass error = env->FindClass("java/lang/IllegalStateException");
    if (error) env->ThrowNew(error, message);
}
}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_linnan_blindassist_hftf_Dav2Yuv420RgbConverter_nativeCreate(JNIEnv*, jobject) {
    try { return reinterpret_cast<jlong>(new Converter()); } catch (...) { return 0; }
}

extern "C" JNIEXPORT void JNICALL
Java_com_linnan_blindassist_hftf_Dav2Yuv420RgbConverter_nativeConvert(
    JNIEnv* env, jobject, jlong handle, jbyteArray y, jbyteArray u, jbyteArray v,
    jint width, jint height, jint rotation_degrees, jbyteArray output) {
    try {
        if (!handle) throw std::runtime_error("YUV converter is closed");
        if (env->GetArrayLength(output) < kOutputWidth * kOutputHeight * 3)
            throw std::runtime_error("RGB array is smaller than the frozen contract");
        jbyte* output_address = env->GetByteArrayElements(output, nullptr);
        if (!output_address) throw std::runtime_error("unable to pin RGB output array");
        try {
            reinterpret_cast<Converter*>(handle)->convert(
                env, y, u, v, width, height, rotation_degrees,
                reinterpret_cast<uint8_t*>(output_address));
        } catch (...) {
            env->ReleaseByteArrayElements(output, output_address, JNI_ABORT);
            throw;
        }
        env->ReleaseByteArrayElements(output, output_address, 0);
    } catch (const std::exception& error) { throwJava(env, error.what()); }
}

extern "C" JNIEXPORT void JNICALL
Java_com_linnan_blindassist_hftf_Dav2Yuv420RgbConverter_nativeConvertDirect(
    JNIEnv* env, jobject, jlong handle, jbyteArray y, jbyteArray u, jbyteArray v,
    jint width, jint height, jint rotation_degrees, jobject output) {
    try {
        if (!handle) throw std::runtime_error("YUV converter is closed");
        auto* output_address = static_cast<uint8_t*>(env->GetDirectBufferAddress(output));
        if (!output_address || env->GetDirectBufferCapacity(output) <
            kOutputWidth * kOutputHeight * 3)
            throw std::runtime_error("invalid direct RGB output buffer");
        reinterpret_cast<Converter*>(handle)->convert(
            env, y, u, v, width, height, rotation_degrees, output_address);
    } catch (const std::exception& error) { throwJava(env, error.what()); }
}

extern "C" JNIEXPORT void JNICALL
Java_com_linnan_blindassist_hftf_Dav2Yuv420RgbConverter_nativeDestroy(
    JNIEnv*, jobject, jlong handle) { delete reinterpret_cast<Converter*>(handle); }

extern "C" JNIEXPORT jboolean JNICALL
Java_com_linnan_blindassist_hftf_Dav2BitmapRgbConverter_nativeConvertBitmap(
    JNIEnv* env, jobject, jobject bitmap, jobject output, jint output_width, jint output_height) {
    if (bitmap == nullptr || output == nullptr || output_width <= 0 || output_height <= 0) {
        return JNI_FALSE;
    }
    AndroidBitmapInfo info{};
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 || info.width == 0 || info.height == 0) {
        return JNI_FALSE;
    }
    auto* destination = static_cast<uint8_t*>(env->GetDirectBufferAddress(output));
    const jlong required = static_cast<jlong>(output_width) * output_height * 3;
    if (destination == nullptr || env->GetDirectBufferCapacity(output) < required) return JNI_FALSE;
    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS || pixels == nullptr) {
        return JNI_FALSE;
    }
    const auto* source = static_cast<const uint8_t*>(pixels);
    for (jint y = 0; y < output_height; ++y) {
        const uint32_t source_y = static_cast<uint32_t>(
            static_cast<uint64_t>(y) * info.height / static_cast<uint32_t>(output_height));
        const auto* source_row = source + static_cast<size_t>(source_y) * info.stride;
        auto* destination_row = destination + static_cast<size_t>(y) * output_width * 3;
        for (jint x = 0; x < output_width; ++x) {
            const uint32_t source_x = static_cast<uint32_t>(
                static_cast<uint64_t>(x) * info.width / static_cast<uint32_t>(output_width));
            const auto* pixel = source_row + static_cast<size_t>(source_x) * 4;
            auto* rgb = destination_row + static_cast<size_t>(x) * 3;
            rgb[0] = pixel[0];
            rgb[1] = pixel[1];
            rgb[2] = pixel[2];
        }
    }
    AndroidBitmap_unlockPixels(env, bitmap);
    return JNI_TRUE;
}
