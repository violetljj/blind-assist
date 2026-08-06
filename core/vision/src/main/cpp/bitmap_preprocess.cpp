#include <jni.h>
#include <android/bitmap.h>
#include <algorithm>
#include <cstdint>

extern "C" JNIEXPORT jboolean JNICALL
Java_com_linnan_blindassist_vision_NativeBitmapPreprocessor_writeArgbToFloatNative(
    JNIEnv* env, jclass, jobject bitmap, jobject output) {
    if (bitmap == nullptr || output == nullptr) return JNI_FALSE;
    AndroidBitmapInfo info{};
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return JNI_FALSE;
    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS || pixels == nullptr) return JNI_FALSE;
    jfloat* floats = static_cast<jfloat*>(env->GetDirectBufferAddress(output));
    const jlong capacity = env->GetDirectBufferCapacity(output);
    const jlong count = static_cast<jlong>(info.width) * info.height;
    if (floats == nullptr || capacity < count * 3 * static_cast<jlong>(sizeof(jfloat))) {
        AndroidBitmap_unlockPixels(env, bitmap);
        return JNI_FALSE;
    }
    const auto* source = static_cast<const uint8_t*>(pixels);
    const float scale = 1.0f / 255.0f;
    for (uint32_t y = 0; y < info.height; ++y) {
        const auto* row = source + static_cast<size_t>(y) * info.stride;
        for (uint32_t x = 0; x < info.width; ++x) {
            const auto* pixel = row + static_cast<size_t>(x) * 4;
            const jlong i = static_cast<jlong>(y) * info.width + x;
            // RGBA_8888 bytes are R,G,B,A on Android.
            floats[i * 3] = static_cast<float>(pixel[0]) * scale;
            floats[i * 3 + 1] = static_cast<float>(pixel[1]) * scale;
            floats[i * 3 + 2] = static_cast<float>(pixel[2]) * scale;
        }
    }
    AndroidBitmap_unlockPixels(env, bitmap);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_linnan_blindassist_vision_NativeBitmapPreprocessor_writePaddedArgbToFloatNative(
    JNIEnv* env, jclass, jobject bitmap, jobject output, jint inputSize, jint top) {
    if (bitmap == nullptr || output == nullptr || inputSize <= 0 || top < 0) return JNI_FALSE;
    AndroidBitmapInfo info{};
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
        info.width != static_cast<uint32_t>(inputSize) ||
        top + static_cast<jint>(info.height) > inputSize) return JNI_FALSE;
    jfloat* floats = static_cast<jfloat*>(env->GetDirectBufferAddress(output));
    const jlong requiredBytes = static_cast<jlong>(inputSize) * inputSize * 3 * sizeof(jfloat);
    if (floats == nullptr || env->GetDirectBufferCapacity(output) < requiredBytes) return JNI_FALSE;
    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS || pixels == nullptr) return JNI_FALSE;
    const jlong count = static_cast<jlong>(inputSize) * inputSize * 3;
    std::fill(floats, floats + count, 0.0f);
    const auto* source = static_cast<const uint8_t*>(pixels);
    constexpr float scale = 1.0f / 255.0f;
    for (uint32_t y = 0; y < info.height; ++y) {
        const auto* row = source + static_cast<size_t>(y) * info.stride;
        const jlong destination = (static_cast<jlong>(top) + y) * inputSize * 3;
        for (uint32_t x = 0; x < info.width; ++x) {
            const auto* pixel = row + static_cast<size_t>(x) * 4;
            const jlong index = destination + static_cast<jlong>(x) * 3;
            floats[index] = pixel[0] * scale;
            floats[index + 1] = pixel[1] * scale;
            floats[index + 2] = pixel[2] * scale;
        }
    }
    AndroidBitmap_unlockPixels(env, bitmap);
    return JNI_TRUE;
}
