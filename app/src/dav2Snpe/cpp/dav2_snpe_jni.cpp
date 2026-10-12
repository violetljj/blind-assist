#include <jni.h>
#include <cstdlib>
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
template<auto Delete> struct Owned {
    void* value = nullptr;
    ~Owned() { if (value) Delete(value); }
};
struct Engine {
    Snpe_SNPE_Handle_t snpe = nullptr;
    Snpe_ITensor_Handle_t input = nullptr;
    std::string input_name, output_name;
    std::vector<jint> shape;
    ~Engine() {
        if (input) Snpe_ITensor_Delete(input);
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
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_linnan_blindassist_vision_Dav2SnpeDepthEstimator_nativeCreate(JNIEnv* env, jobject, jstring path, jstring dsp_path) {
    try {
        const auto dsp = string(env, dsp_path);
        const char* previous = std::getenv("ADSP_LIBRARY_PATH");
        std::string search = dsp + ";" + (previous ? previous : "/vendor/lib/rfsa/adsp;/vendor/dsp;/system/lib/rfsa/adsp;/dsp");
        require(setenv("ADSP_LIBRARY_PATH", search.c_str(), 1) == 0, "Set DSP search path");
        require(Snpe_Util_IsRuntimeAvailableCheckOption(SNPE_RUNTIME_DSP, SNPE_RUNTIME_CHECK_OPTION_UNSIGNEDPD_CHECK), "HTP runtime unavailable");
        Owned<Snpe_DlContainer_Delete> container{Snpe_DlContainer_Open(string(env, path).c_str())};
        require(container.value, "Open DLC");
        Owned<Snpe_SNPEBuilder_Delete> builder{Snpe_SNPEBuilder_Create(container.value)};
        Owned<Snpe_RuntimeList_Delete> runtimes{Snpe_RuntimeList_Create()};
        Owned<Snpe_PlatformConfig_Delete> platform{Snpe_PlatformConfig_Create()};
        require(builder.value && runtimes.value && platform.value, "Create SNPE builder");
        require(Snpe_RuntimeList_Add(runtimes.value, SNPE_RUNTIME_DSP) == SNPE_SUCCESS, "DSP runtime order");
        require(Snpe_PlatformConfig_SetPlatformOptions(platform.value, "unsignedPD:ON"), "Unsigned DSP platform");
        require(Snpe_SNPEBuilder_SetRuntimeProcessorOrder(builder.value, runtimes.value) == SNPE_SUCCESS, "Set DSP-only runtime");
        require(Snpe_SNPEBuilder_SetPlatformConfig(builder.value, platform.value) == SNPE_SUCCESS, "Set platform");
        require(Snpe_SNPEBuilder_SetPerformanceProfile(builder.value, SNPE_PERFORMANCE_PROFILE_BURST) == SNPE_SUCCESS, "Set burst");
        require(Snpe_SNPEBuilder_SetUseUserSuppliedBuffers(builder.value, false) == SNPE_SUCCESS, "Set ITensor");
        auto engine = std::make_unique<Engine>();
        engine->snpe = Snpe_SNPEBuilder_Build(builder.value);
        require(engine->snpe, "Build DSP-only DAV2");
        Owned<Snpe_StringList_Delete> inputs{Snpe_SNPE_GetInputTensorNames(engine->snpe)};
        Owned<Snpe_StringList_Delete> outputs{Snpe_SNPE_GetOutputTensorNames(engine->snpe)};
        require(inputs.value && outputs.value && Snpe_StringList_Size(inputs.value) == 1 && Snpe_StringList_Size(outputs.value) == 1, "One input/output required");
        engine->input_name = Snpe_StringList_At(inputs.value, 0);
        engine->output_name = Snpe_StringList_At(outputs.value, 0);
        Owned<Snpe_TensorShape_Delete> shape{Snpe_SNPE_GetInputDimensions(engine->snpe, engine->input_name.c_str())};
        require(shape.value, "Input shape");
        for (size_t i = 0; i < Snpe_TensorShape_Rank(shape.value); ++i)
            engine->shape.push_back(static_cast<jint>(Snpe_TensorShape_At(shape.value, i)));
        engine->input = Snpe_Util_CreateITensor(shape.value);
        require(engine->input && Snpe_ITensor_GetSize(engine->input) == 686 * 518 * 3, "Fixed DAV2 input");
        return reinterpret_cast<jlong>(engine.release());
    } catch (const std::exception& error) { fail(env, error); return 0; }
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_linnan_blindassist_vision_Dav2SnpeDepthEstimator_nativeShape(JNIEnv* env, jobject, jlong handle) {
    auto* engine = reinterpret_cast<Engine*>(handle);
    auto result = env->NewIntArray(engine->shape.size());
    if (result) env->SetIntArrayRegion(result, 0, engine->shape.size(), engine->shape.data());
    return result;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_linnan_blindassist_vision_Dav2SnpeDepthEstimator_nativeExecute(JNIEnv* env, jobject, jlong handle, jfloatArray input) {
    try {
        auto* engine = reinterpret_cast<Engine*>(handle);
        require(engine && env->GetArrayLength(input) == 686 * 518 * 3, "Input size");
        env->GetFloatArrayRegion(input, 0, 686 * 518 * 3, static_cast<float*>(Snpe_ITensor_GetData(engine->input)));
        if (env->ExceptionCheck()) return nullptr;
        Owned<Snpe_TensorMap_Delete> inputs{Snpe_TensorMap_Create()}, outputs{Snpe_TensorMap_Create()};
        require(inputs.value && outputs.value, "Create tensor maps");
        Snpe_TensorMap_Add(inputs.value, engine->input_name.c_str(), engine->input);
        require(Snpe_SNPE_ExecuteITensors(engine->snpe, inputs.value, outputs.value) == SNPE_SUCCESS, "Execute DSP DAV2");
        auto tensor = Snpe_TensorMap_GetTensor_Ref(outputs.value, engine->output_name.c_str());
        require(tensor && Snpe_ITensor_GetSize(tensor) == 686 * 518, "Output size");
        auto result = env->NewFloatArray(686 * 518);
        if (result) env->SetFloatArrayRegion(result, 0, 686 * 518, static_cast<float*>(Snpe_ITensor_GetData(tensor)));
        return result;
    } catch (const std::exception& error) { fail(env, error); return nullptr; }
}

extern "C" JNIEXPORT void JNICALL
Java_com_linnan_blindassist_vision_Dav2SnpeDepthEstimator_nativeDestroy(JNIEnv*, jobject, jlong handle) {
    delete reinterpret_cast<Engine*>(handle);
}
