#include <jni.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
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

template <auto Delete>
struct Owned {
    void* value = nullptr;
    ~Owned() {
        if (value) Delete(value);
    }
};

struct EgoClearHtpEngine {
    Snpe_SNPE_Handle_t snpe = nullptr;
    Snpe_ITensor_Handle_t t_rgb = nullptr;
    Snpe_ITensor_Handle_t t_k = nullptr;
    Snpe_ITensor_Handle_t t_r = nullptr;
    size_t n_rgb = 0;
    size_t n_k = 0;
    size_t n_r = 0;
    std::string aux_name = "disp_cues";

    ~EgoClearHtpEngine() {
        if (t_rgb) Snpe_ITensor_Delete(t_rgb);
        if (t_k) Snpe_ITensor_Delete(t_k);
        if (t_r) Snpe_ITensor_Delete(t_r);
        if (snpe) Snpe_SNPE_Delete(snpe);
    }
};

struct EgoClearHtpStepEngine {
    Snpe_SNPE_Handle_t snpe = nullptr;
    std::vector<std::string> in_names;
    std::vector<Snpe_ITensor_Handle_t> in_tensors;
    std::vector<size_t> in_sizes;
    std::string corr_out_name = "corr_probs";
    bool has_omni = false;

    ~EgoClearHtpStepEngine() {
        for (auto t : in_tensors) {
            if (t) Snpe_ITensor_Delete(t);
        }
        if (snpe) Snpe_SNPE_Delete(snpe);
    }
};

void fail(JNIEnv* env, const std::exception& error) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error.what());
}

std::string jstr(JNIEnv* env, jstring value) {
    const char* p = env->GetStringUTFChars(value, nullptr);
    if (!p) throw std::runtime_error("Unable to read JNI string");
    std::string out(p);
    env->ReleaseStringUTFChars(value, p);
    return out;
}

Snpe_ITensor_Handle_t createInput(Snpe_SNPE_Handle_t snpe, const char* name, size_t expected) {
    Owned<Snpe_TensorShape_Delete> shape{Snpe_SNPE_GetInputDimensions(snpe, name)};
    require(shape.value, name);
    auto t = Snpe_Util_CreateITensor(shape.value);
    require(t && (expected == 0 || Snpe_ITensor_GetSize(t) == expected), "Input tensor size mismatch");
    return t;
}

void ensureDspRuntime(const std::string& dsp) {
    const char* prev = std::getenv("ADSP_LIBRARY_PATH");
    std::string search = dsp + ";" + (prev ? prev : "/vendor/lib/rfsa/adsp;/vendor/dsp;/system/lib/rfsa/adsp;/dsp");
    require(setenv("ADSP_LIBRARY_PATH", search.c_str(), 1) == 0, "Set ADSP_LIBRARY_PATH");
    require(
        Snpe_Util_IsRuntimeAvailableCheckOption(SNPE_RUNTIME_DSP, SNPE_RUNTIME_CHECK_OPTION_UNSIGNEDPD_CHECK),
        "Hexagon HTP runtime unavailable");
}

jlong createEngineImpl(JNIEnv* env, const std::string& dlc, const std::string& dsp, const std::string& aux) {
    ensureDspRuntime(dsp);

    Owned<Snpe_DlContainer_Delete> container{Snpe_DlContainer_Open(dlc.c_str())};
    require(container.value, "Open EgoClear DLC");

    Owned<Snpe_SNPEBuilder_Delete> builder{Snpe_SNPEBuilder_Create(container.value)};
    Owned<Snpe_RuntimeList_Delete> runtimes{Snpe_RuntimeList_Create()};
    Owned<Snpe_PlatformConfig_Delete> platform{Snpe_PlatformConfig_Create()};
    Owned<Snpe_StringList_Delete> out_names{Snpe_StringList_Create()};
    require(builder.value && runtimes.value && platform.value && out_names.value, "Create SNPE objects");

    require(Snpe_RuntimeList_Add(runtimes.value, SNPE_RUNTIME_DSP) == SNPE_SUCCESS, "Add DSP runtime");
    require(Snpe_PlatformConfig_SetPlatformOptions(platform.value, "unsignedPD:ON"), "Set unsignedPD");
    require(Snpe_SNPEBuilder_SetRuntimeProcessorOrder(builder.value, runtimes.value) == SNPE_SUCCESS, "Set DSP-only order");
    require(Snpe_SNPEBuilder_SetPlatformConfig(builder.value, platform.value) == SNPE_SUCCESS, "Set platform config");
    require(Snpe_SNPEBuilder_SetPerformanceProfile(builder.value, SNPE_PERFORMANCE_PROFILE_BURST) == SNPE_SUCCESS, "Set burst profile");
    require(Snpe_SNPEBuilder_SetUseUserSuppliedBuffers(builder.value, false) == SNPE_SUCCESS, "Set ITensor mode");
    require(Snpe_StringList_Append(out_names.value, "f2d") == SNPE_SUCCESS, "Append f2d");
    require(Snpe_StringList_Append(out_names.value, aux.c_str()) == SNPE_SUCCESS, "Append aux output");
    require(Snpe_SNPEBuilder_SetOutputTensors(builder.value, out_names.value) == SNPE_SUCCESS, "Set output tensors");

    auto engine = std::make_unique<EgoClearHtpEngine>();
    engine->aux_name = aux;
    engine->snpe = Snpe_SNPEBuilder_Build(builder.value);
    require(engine->snpe, "Build DSP-only EgoClear HTP engine");

    engine->n_rgb = 3 * 294 * 518;
    engine->n_k = 4;
    engine->n_r = 9;
    engine->t_rgb = createInput(engine->snpe, "rgb", engine->n_rgb);
    engine->t_k = createInput(engine->snpe, "K_scaled", engine->n_k);
    engine->t_r = createInput(engine->snpe, "R_c2w", engine->n_r);
    return reinterpret_cast<jlong>(engine.release());
}

jlong createStepImpl(
    JNIEnv*,
    const std::string& dlc,
    const std::string& dsp,
    const std::string& aux_prefix,
    bool has_omni) {
    ensureDspRuntime(dsp);

    Owned<Snpe_DlContainer_Delete> container{Snpe_DlContainer_Open(dlc.c_str())};
    require(container.value, "Open EgoClear Step DLC");

    Owned<Snpe_SNPEBuilder_Delete> builder{Snpe_SNPEBuilder_Create(container.value)};
    Owned<Snpe_RuntimeList_Delete> runtimes{Snpe_RuntimeList_Create()};
    Owned<Snpe_PlatformConfig_Delete> platform{Snpe_PlatformConfig_Create()};
    Owned<Snpe_StringList_Delete> out_names{Snpe_StringList_Create()};
    require(builder.value && runtimes.value && platform.value && out_names.value, "Create SNPE step objects");

    require(Snpe_RuntimeList_Add(runtimes.value, SNPE_RUNTIME_DSP) == SNPE_SUCCESS, "Add DSP runtime");
    require(Snpe_PlatformConfig_SetPlatformOptions(platform.value, "unsignedPD:ON"), "Set unsignedPD");
    require(Snpe_SNPEBuilder_SetRuntimeProcessorOrder(builder.value, runtimes.value) == SNPE_SUCCESS, "Set DSP-only order");
    require(Snpe_SNPEBuilder_SetPlatformConfig(builder.value, platform.value) == SNPE_SUCCESS, "Set platform config");
    require(Snpe_SNPEBuilder_SetPerformanceProfile(builder.value, SNPE_PERFORMANCE_PROFILE_BURST) == SNPE_SUCCESS, "Set burst profile");
    require(Snpe_SNPEBuilder_SetUseUserSuppliedBuffers(builder.value, false) == SNPE_SUCCESS, "Set ITensor mode");

    const std::string corr_name = (aux_prefix == "dc") ? "corr_probs_monotonic" : "corr_probs";
    const char* base_outs[] = {"static_logits", "dynamic_logits", corr_name.c_str(), "unc_logits", "mem_new", "age_new"};
    for (const char* o : base_outs) {
        require(Snpe_StringList_Append(out_names.value, o) == SNPE_SUCCESS, o);
    }
    if (has_omni) {
        require(Snpe_StringList_Append(out_names.value, "omni_probs") == SNPE_SUCCESS, "omni_probs");
    }
    require(Snpe_SNPEBuilder_SetOutputTensors(builder.value, out_names.value) == SNPE_SUCCESS, "Set step outputs");

    auto engine = std::make_unique<EgoClearHtpStepEngine>();
    engine->corr_out_name = corr_name;
    engine->has_omni = has_omni;
    engine->snpe = Snpe_SNPEBuilder_Build(builder.value);
    require(engine->snpe, "Build DSP-only EgoClear HTP step engine");

    engine->in_names = {
        "f2d_t", aux_prefix + "_t",
        "f2d_1", aux_prefix + "_1",
        "f2d_2", aux_prefix + "_2",
        "K_t", "R_t", "t_t", "fwd_t",
        "K_1", "R_1", "t_1", "fwd_1",
        "K_2", "R_2", "t_2", "ds_t",
        "mem_prev", "age_prev",
    };
    engine->in_tensors.reserve(engine->in_names.size());
    engine->in_sizes.reserve(engine->in_names.size());
    for (const auto& name : engine->in_names) {
        auto t = createInput(engine->snpe, name.c_str(), 0);
        engine->in_tensors.push_back(t);
        engine->in_sizes.push_back(Snpe_ITensor_GetSize(t));
    }
    return reinterpret_cast<jlong>(engine.release());
}

void copyOutTensor(JNIEnv* env, Snpe_TensorMap_Handle_t outputs, const char* name, jfloatArray dst) {
    auto t = Snpe_TensorMap_GetTensor_Ref(outputs, name);
    require(t != nullptr, name);
    const size_t n = Snpe_ITensor_GetSize(t);
    require(static_cast<size_t>(env->GetArrayLength(dst)) == n, name);
    env->SetFloatArrayRegion(dst, 0, static_cast<jsize>(n), static_cast<float*>(Snpe_ITensor_GetData(t)));
}

struct ImageNetLut {
    float r[256];
    float g[256];
    float b[256];
    ImageNetLut() {
        constexpr float invR = 1.0f / (255.0f * 0.229f);
        constexpr float invG = 1.0f / (255.0f * 0.224f);
        constexpr float invB = 1.0f / (255.0f * 0.225f);
        constexpr float offR = 0.485f / 0.229f;
        constexpr float offG = 0.456f / 0.224f;
        constexpr float offB = 0.406f / 0.225f;
        for (int i = 0; i < 256; ++i) {
            r[i] = i * invR - offR;
            g[i] = i * invG - offG;
            b[i] = i * invB - offB;
        }
    }
};

const ImageNetLut& getImageNetLut() {
    static const ImageNetLut kLut;
    return kLut;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_linnan_blindassist_egoclear_EgoClearSnpeEncoder_nativeCreate(
    JNIEnv* env, jobject, jstring dlc_path, jstring dsp_dir) {
    try {
        return createEngineImpl(env, jstr(env, dlc_path), jstr(env, dsp_dir), "disp_cues");
    } catch (const std::exception& error) {
        fail(env, error);
        return 0;
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_linnan_blindassist_egoclear_EgoClearSnpeEncoder_nativeCreateWithAux(
    JNIEnv* env, jobject, jstring dlc_path, jstring dsp_dir, jstring aux_output_name) {
    try {
        return createEngineImpl(env, jstr(env, dlc_path), jstr(env, dsp_dir), jstr(env, aux_output_name));
    } catch (const std::exception& error) {
        fail(env, error);
        return 0;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_linnan_blindassist_egoclear_EgoClearSnpeEncoder_nativeExecute(
    JNIEnv* env, jobject, jlong handle,
    jfloatArray rgb, jfloatArray k_scaled, jfloatArray r_c2w,
    jfloatArray out_f2d, jfloatArray out_dc) {
    try {
        auto* engine = reinterpret_cast<EgoClearHtpEngine*>(handle);
        require(engine != nullptr, "Null EgoClear HTP handle");
        require(static_cast<size_t>(env->GetArrayLength(rgb)) == engine->n_rgb, "rgb size");
        require(static_cast<size_t>(env->GetArrayLength(k_scaled)) == engine->n_k, "K_scaled size");
        require(static_cast<size_t>(env->GetArrayLength(r_c2w)) == engine->n_r, "R_c2w size");

        env->GetFloatArrayRegion(rgb, 0, static_cast<jsize>(engine->n_rgb), static_cast<float*>(Snpe_ITensor_GetData(engine->t_rgb)));
        env->GetFloatArrayRegion(k_scaled, 0, static_cast<jsize>(engine->n_k), static_cast<float*>(Snpe_ITensor_GetData(engine->t_k)));
        env->GetFloatArrayRegion(r_c2w, 0, static_cast<jsize>(engine->n_r), static_cast<float*>(Snpe_ITensor_GetData(engine->t_r)));
        if (env->ExceptionCheck()) return;

        Owned<Snpe_TensorMap_Delete> inputs{Snpe_TensorMap_Create()};
        Owned<Snpe_TensorMap_Delete> outputs{Snpe_TensorMap_Create()};
        require(inputs.value && outputs.value, "Create tensor maps");
        Snpe_TensorMap_Add(inputs.value, "rgb", engine->t_rgb);
        Snpe_TensorMap_Add(inputs.value, "K_scaled", engine->t_k);
        Snpe_TensorMap_Add(inputs.value, "R_c2w", engine->t_r);

        require(Snpe_SNPE_ExecuteITensors(engine->snpe, inputs.value, outputs.value) == SNPE_SUCCESS, "Execute EgoClear HTP encoder");

        copyOutTensor(env, outputs.value, "f2d", out_f2d);
        copyOutTensor(env, outputs.value, engine->aux_name.c_str(), out_dc);
    } catch (const std::exception& error) {
        fail(env, error);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_linnan_blindassist_egoclear_EgoClearSnpeEncoder_nativeDestroy(JNIEnv*, jobject, jlong handle) {
    delete reinterpret_cast<EgoClearHtpEngine*>(handle);
}

// ============================================================================
// EgoClearSnpeStep — Full-NPU Hexagon v75 20-input streaming step (~4.9 ms!)
// ============================================================================

extern "C" JNIEXPORT jlong JNICALL
Java_com_linnan_blindassist_egoclear_EgoClearSnpeStep_nativeCreateStep(
    JNIEnv* env, jobject, jstring dlc_path, jstring dsp_dir, jstring aux_prefix, jboolean has_omni) {
    try {
        return createStepImpl(env, jstr(env, dlc_path), jstr(env, dsp_dir), jstr(env, aux_prefix), has_omni == JNI_TRUE);
    } catch (const std::exception& error) {
        fail(env, error);
        return 0;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_linnan_blindassist_egoclear_EgoClearSnpeStep_nativeExecuteStep(
    JNIEnv* env, jobject, jlong handle,
    jobjectArray in_arrays,
    jfloatArray out_static, jfloatArray out_dynamic, jfloatArray out_corr,
    jfloatArray out_unc, jfloatArray out_omni, jfloatArray out_mem, jfloatArray out_age) {
    try {
        auto* engine = reinterpret_cast<EgoClearHtpStepEngine*>(handle);
        require(engine != nullptr, "Null EgoClear HTP step handle");
        const jsize n_in = env->GetArrayLength(in_arrays);
        require(static_cast<size_t>(n_in) == engine->in_tensors.size(), "Step input array count mismatch");

        Owned<Snpe_TensorMap_Delete> inputs{Snpe_TensorMap_Create()};
        Owned<Snpe_TensorMap_Delete> outputs{Snpe_TensorMap_Create()};
        require(inputs.value && outputs.value, "Create step tensor maps");

        for (jsize i = 0; i < n_in; ++i) {
            auto arr = reinterpret_cast<jfloatArray>(env->GetObjectArrayElement(in_arrays, i));
            require(arr != nullptr, "Null step input array");
            const size_t expected = engine->in_sizes[i];
            require(static_cast<size_t>(env->GetArrayLength(arr)) == expected, engine->in_names[i].c_str());
            env->GetFloatArrayRegion(arr, 0, static_cast<jsize>(expected), static_cast<float*>(Snpe_ITensor_GetData(engine->in_tensors[i])));
            env->DeleteLocalRef(arr);
            if (env->ExceptionCheck()) return;
            Snpe_TensorMap_Add(inputs.value, engine->in_names[i].c_str(), engine->in_tensors[i]);
        }

        require(Snpe_SNPE_ExecuteITensors(engine->snpe, inputs.value, outputs.value) == SNPE_SUCCESS, "Execute EgoClear HTP step");

        copyOutTensor(env, outputs.value, "static_logits", out_static);
        copyOutTensor(env, outputs.value, "dynamic_logits", out_dynamic);
        copyOutTensor(env, outputs.value, engine->corr_out_name.c_str(), out_corr);
        copyOutTensor(env, outputs.value, "unc_logits", out_unc);
        copyOutTensor(env, outputs.value, "mem_new", out_mem);
        copyOutTensor(env, outputs.value, "age_new", out_age);
        if (engine->has_omni && out_omni != nullptr) {
            copyOutTensor(env, outputs.value, "omni_probs", out_omni);
        }
    } catch (const std::exception& error) {
        fail(env, error);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_linnan_blindassist_egoclear_EgoClearSnpeStep_nativeDestroyStep(JNIEnv*, jobject, jlong handle) {
    delete reinterpret_cast<EgoClearHtpStepEngine*>(handle);
}

// ============================================================================
// EgoClearNativePreproc — Fast C++ DirectBuffer YUV420 -> ImageNet NCHW (~1ms)
// ============================================================================

extern "C" JNIEXPORT jfloat JNICALL
Java_com_linnan_blindassist_egoclear_EgoClearNativePreproc_nativePreprocessYuv420ToNchw(
    JNIEnv* env, jobject,
    jobject y_buf, jint y_row_stride, jint y_pix_stride,
    jobject u_buf, jint u_row_stride, jint u_pix_stride,
    jobject v_buf, jint v_row_stride, jint v_pix_stride,
    jint src_w, jint src_h, jint rot_deg,
    jfloat crop_x0, jfloat crop_y0, jfloat crop_w, jfloat crop_h,
    jint out_w, jint out_h, jfloatArray out_nchw) {
    const auto* y_ptr = static_cast<const uint8_t*>(env->GetDirectBufferAddress(y_buf));
    const auto* u_ptr = static_cast<const uint8_t*>(env->GetDirectBufferAddress(u_buf));
    const auto* v_ptr = static_cast<const uint8_t*>(env->GetDirectBufferAddress(v_buf));
    if (!y_ptr || !u_ptr || !v_ptr || out_w <= 0 || out_h <= 0 || src_w <= 0 || src_h <= 0) {
        return -1.0f;
    }
    const int plane = out_w * out_h;
    if (env->GetArrayLength(out_nchw) < 3 * plane) {
        return -1.0f;
    }

    const int max_sx = src_w - 1;
    const int max_sy = src_h - 1;
    const float scale_x = crop_w / static_cast<float>(out_w);
    const float scale_y = crop_h / static_cast<float>(out_h);

    std::vector<int> ix_lut(out_w);
    std::vector<int> iy_lut(out_h);
    for (int ox = 0; ox < out_w; ++ox) {
        ix_lut[ox] = static_cast<int>(crop_x0 + (static_cast<float>(ox) + 0.5f) * scale_x);
    }
    for (int oy = 0; oy < out_h; ++oy) {
        iy_lut[oy] = static_cast<int>(crop_y0 + (static_cast<float>(oy) + 0.5f) * scale_y);
    }

    const auto& lut = getImageNetLut();
    auto* out = static_cast<float*>(env->GetPrimitiveArrayCritical(out_nchw, nullptr));
    if (!out) return -1.0f;
    float* __restrict out_r = out;
    float* __restrict out_g = out + plane;
    float* __restrict out_b = out + 2 * plane;

    int64_t luma_sum = 0;
    auto clamp_i = [](int v, int hi) -> int { return v < 0 ? 0 : (v > hi ? hi : v); };
    auto clamp_u8 = [](int v) -> int { return v < 0 ? 0 : (v > 255 ? 255 : v); };

    if (rot_deg == 90 || rot_deg == 270) {
        std::vector<int> sy_col(out_w);
        std::vector<int> y_row_off(out_w);
        std::vector<int> u_row_off(out_w);
        std::vector<int> v_row_off(out_w);
        for (int ox = 0; ox < out_w; ++ox) {
            const int sy = (rot_deg == 90) ? clamp_i(max_sy - ix_lut[ox], max_sy) : clamp_i(ix_lut[ox], max_sy);
            sy_col[ox] = sy;
            y_row_off[ox] = sy * y_row_stride;
            const int uv_y = sy >> 1;
            u_row_off[ox] = uv_y * u_row_stride;
            v_row_off[ox] = uv_y * v_row_stride;
        }
        int out_idx = 0;
        for (int oy = 0; oy < out_h; ++oy) {
            const int sx = (rot_deg == 90) ? clamp_i(iy_lut[oy], max_sx) : clamp_i(max_sx - iy_lut[oy], max_sx);
            const int y_xoff = sx * y_pix_stride;
            const int uv_x = sx >> 1;
            const int u_xoff = uv_x * u_pix_stride;
            const int v_xoff = uv_x * v_pix_stride;
            for (int ox = 0; ox < out_w; ++ox) {
                const int yv = y_ptr[y_row_off[ox] + y_xoff];
                luma_sum += yv;
                const int uv = static_cast<int>(u_ptr[u_row_off[ox] + u_xoff]) - 128;
                const int vv = static_cast<int>(v_ptr[v_row_off[ox] + v_xoff]) - 128;
                const int r = clamp_u8(yv + ((1436 * vv) >> 10));
                const int g = clamp_u8(yv - ((352 * uv + 731 * vv) >> 10));
                const int b = clamp_u8(yv + ((1815 * uv) >> 10));
                out_r[out_idx] = lut.r[r];
                out_g[out_idx] = lut.g[g];
                out_b[out_idx] = lut.b[b];
                ++out_idx;
            }
        }
    } else {
        std::vector<int> y_xoff(out_w);
        std::vector<int> u_xoff(out_w);
        std::vector<int> v_xoff(out_w);
        for (int ox = 0; ox < out_w; ++ox) {
            const int sx = (rot_deg == 180) ? clamp_i(max_sx - ix_lut[ox], max_sx) : clamp_i(ix_lut[ox], max_sx);
            y_xoff[ox] = sx * y_pix_stride;
            const int uv_x = sx >> 1;
            u_xoff[ox] = uv_x * u_pix_stride;
            v_xoff[ox] = uv_x * v_pix_stride;
        }
        int out_idx = 0;
        for (int oy = 0; oy < out_h; ++oy) {
            const int sy = (rot_deg == 180) ? clamp_i(max_sy - iy_lut[oy], max_sy) : clamp_i(iy_lut[oy], max_sy);
            const uint8_t* y_row = y_ptr + sy * y_row_stride;
            const int uv_y = sy >> 1;
            const uint8_t* u_row = u_ptr + uv_y * u_row_stride;
            const uint8_t* v_row = v_ptr + uv_y * v_row_stride;
            for (int ox = 0; ox < out_w; ++ox) {
                const int yv = y_row[y_xoff[ox]];
                luma_sum += yv;
                const int uv = static_cast<int>(u_row[u_xoff[ox]]) - 128;
                const int vv = static_cast<int>(v_row[v_xoff[ox]]) - 128;
                const int r = clamp_u8(yv + ((1436 * vv) >> 10));
                const int g = clamp_u8(yv - ((352 * uv + 731 * vv) >> 10));
                const int b = clamp_u8(yv + ((1815 * uv) >> 10));
                out_r[out_idx] = lut.r[r];
                out_g[out_idx] = lut.g[g];
                out_b[out_idx] = lut.b[b];
                ++out_idx;
            }
        }
    }

    env->ReleasePrimitiveArrayCritical(out_nchw, out, 0);
    return static_cast<jfloat>(static_cast<double>(luma_sum) / static_cast<double>(plane));
}
