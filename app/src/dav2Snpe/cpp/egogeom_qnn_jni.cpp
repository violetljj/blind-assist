#include <android/log.h>
#include <dlfcn.h>
#include <jni.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "EgoGeomQnn", __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, "EgoGeomQnn", __VA_ARGS__)

namespace {

using QnnInterfaceGetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
using QnnSystemInterfaceGetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);

void requireQnn(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

std::string jstringToStd(JNIEnv* env, jstring s) {
    const char* p = env->GetStringUTFChars(s, nullptr);
    if (!p) throw std::runtime_error("Unable to read JNI string");
    std::string out(p);
    env->ReleaseStringUTFChars(s, p);
    return out;
}

void throwIllegalState(JNIEnv* env, const std::exception& e) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), e.what());
}

struct TensorSlot {
    std::string name;
    Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
    std::vector<uint32_t> dims;
    std::vector<float> buf;
    size_t numElements = 0;
};

static size_t productDims(const uint32_t* dims, uint32_t rank) {
    size_t n = 1;
    for (uint32_t i = 0; i < rank; ++i) n *= static_cast<size_t>(dims[i]);
    return n;
}

static TensorSlot cloneTensorMeta(const Qnn_Tensor_t& src) {
    TensorSlot slot;
    slot.tensor = src;
    if (src.version == QNN_TENSOR_VERSION_1) {
        const auto& v1 = src.v1;
        slot.name = v1.name ? v1.name : "";
        slot.dims.assign(v1.dimensions, v1.dimensions + v1.rank);
        slot.numElements = productDims(slot.dims.data(), v1.rank);
        requireQnn(v1.dataType == QNN_DATATYPE_FLOAT_32,
                   "Expected FLOAT_32 IO tensor for " + slot.name + " (preserve_io_datatype)");
        slot.buf.assign(slot.numElements, 0.0f);
        slot.tensor.v1.name = slot.name.c_str();
        slot.tensor.v1.dimensions = slot.dims.data();
        slot.tensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
        slot.tensor.v1.clientBuf.data = slot.buf.data();
        slot.tensor.v1.clientBuf.dataSize = static_cast<uint32_t>(slot.numElements * sizeof(float));
    } else if (src.version == QNN_TENSOR_VERSION_2) {
        const auto& v2 = src.v2;
        slot.name = v2.name ? v2.name : "";
        slot.dims.assign(v2.dimensions, v2.dimensions + v2.rank);
        slot.numElements = productDims(slot.dims.data(), v2.rank);
        requireQnn(v2.dataType == QNN_DATATYPE_FLOAT_32,
                   "Expected FLOAT_32 IO tensor for " + slot.name + " (preserve_io_datatype)");
        slot.buf.assign(slot.numElements, 0.0f);
        slot.tensor.v2.name = slot.name.c_str();
        slot.tensor.v2.dimensions = slot.dims.data();
        slot.tensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
        slot.tensor.v2.clientBuf.data = slot.buf.data();
        slot.tensor.v2.clientBuf.dataSize = static_cast<uint32_t>(slot.numElements * sizeof(float));
    } else {
        throw std::runtime_error("Unsupported Qnn_TensorVersion_t: " + std::to_string(src.version));
    }
    return slot;
}

struct EgoGeomQnnEngine {
    void* backendLib = nullptr;
    void* sysLib = nullptr;
    QNN_INTERFACE_VER_TYPE qnn{};
    QNN_SYSTEM_INTERFACE_VER_TYPE qnnSys{};

    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    Qnn_ContextHandle_t context = nullptr;
    Qnn_GraphHandle_t graph = nullptr;
    const QnnHtpDevice_PerfInfrastructure_t* htpPerf = nullptr;
    uint32_t powerConfigId = 0;
    bool hasPowerConfig = false;

    std::string graphName;
    std::vector<TensorSlot> inputs;
    std::vector<TensorSlot> outputs;
    std::vector<Qnn_Tensor_t> qnnInputs;
    std::vector<Qnn_Tensor_t> qnnOutputs;
    std::unordered_map<std::string, size_t> inByName;
    std::unordered_map<std::string, size_t> outByName;

    void refreshTensorArrays() {
        qnnInputs.resize(inputs.size());
        inByName.clear();
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto& s = inputs[i];
            if (s.tensor.version == QNN_TENSOR_VERSION_1) {
                s.tensor.v1.name = s.name.c_str();
                s.tensor.v1.dimensions = s.dims.data();
                s.tensor.v1.clientBuf.data = s.buf.data();
                s.tensor.v1.clientBuf.dataSize = static_cast<uint32_t>(s.buf.size() * sizeof(float));
            } else {
                s.tensor.v2.name = s.name.c_str();
                s.tensor.v2.dimensions = s.dims.data();
                s.tensor.v2.clientBuf.data = s.buf.data();
                s.tensor.v2.clientBuf.dataSize = static_cast<uint32_t>(s.buf.size() * sizeof(float));
            }
            qnnInputs[i] = s.tensor;
            inByName[s.name] = i;
        }
        qnnOutputs.resize(outputs.size());
        outByName.clear();
        for (size_t i = 0; i < outputs.size(); ++i) {
            auto& s = outputs[i];
            if (s.tensor.version == QNN_TENSOR_VERSION_1) {
                s.tensor.v1.name = s.name.c_str();
                s.tensor.v1.dimensions = s.dims.data();
                s.tensor.v1.clientBuf.data = s.buf.data();
                s.tensor.v1.clientBuf.dataSize = static_cast<uint32_t>(s.buf.size() * sizeof(float));
            } else {
                s.tensor.v2.name = s.name.c_str();
                s.tensor.v2.dimensions = s.dims.data();
                s.tensor.v2.clientBuf.data = s.buf.data();
                s.tensor.v2.clientBuf.dataSize = static_cast<uint32_t>(s.buf.size() * sizeof(float));
            }
            qnnOutputs[i] = s.tensor;
            outByName[s.name] = i;
        }
    }

    ~EgoGeomQnnEngine() {
        if (hasPowerConfig && htpPerf && htpPerf->destroyPowerConfigId) {
            htpPerf->destroyPowerConfigId(powerConfigId);
        }
        if (context && qnn.contextFree) {
            qnn.contextFree(context, nullptr);
        }
        if (device && qnn.deviceFree) {
            qnn.deviceFree(device);
        }
        if (backend && qnn.backendFree) {
            qnn.backendFree(backend);
        }
        if (sysLib) dlclose(sysLib);
        if (backendLib) dlclose(backendLib);
    }
};

void extractFirstGraphMeta(const QnnSystemContext_BinaryInfo_t* info,
                           std::string& graphName,
                           std::vector<TensorSlot>& inputs,
                           std::vector<TensorSlot>& outputs) {
    requireQnn(info != nullptr, "Null QnnSystemContext_BinaryInfo_t");
    uint32_t numGraphs = 0;
    const QnnSystemContext_GraphInfo_t* graphs = nullptr;
    if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
        numGraphs = info->contextBinaryInfoV1.numGraphs;
        graphs = info->contextBinaryInfoV1.graphs;
    } else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        numGraphs = info->contextBinaryInfoV2.numGraphs;
        graphs = info->contextBinaryInfoV2.graphs;
    } else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
        numGraphs = info->contextBinaryInfoV3.numGraphs;
        graphs = info->contextBinaryInfoV3.graphs;
    } else {
        throw std::runtime_error("Unsupported BinaryInfo version: " + std::to_string(info->version));
    }
    requireQnn(numGraphs >= 1 && graphs != nullptr, "Context binary has no graphs");

    const auto& g0 = graphs[0];
    const char* name = nullptr;
    uint32_t nIn = 0, nOut = 0;
    const Qnn_Tensor_t* inT = nullptr;
    const Qnn_Tensor_t* outT = nullptr;
    if (g0.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        name = g0.graphInfoV1.graphName;
        nIn = g0.graphInfoV1.numGraphInputs;
        inT = g0.graphInfoV1.graphInputs;
        nOut = g0.graphInfoV1.numGraphOutputs;
        outT = g0.graphInfoV1.graphOutputs;
    } else if (g0.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
        name = g0.graphInfoV2.graphName;
        nIn = g0.graphInfoV2.numGraphInputs;
        inT = g0.graphInfoV2.graphInputs;
        nOut = g0.graphInfoV2.numGraphOutputs;
        outT = g0.graphInfoV2.graphOutputs;
    } else if (g0.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        name = g0.graphInfoV3.graphName;
        nIn = g0.graphInfoV3.numGraphInputs;
        inT = g0.graphInfoV3.graphInputs;
        nOut = g0.graphInfoV3.numGraphOutputs;
        outT = g0.graphInfoV3.graphOutputs;
    } else {
        throw std::runtime_error("Unsupported GraphInfo version: " + std::to_string(g0.version));
    }
    requireQnn(name != nullptr && nIn >= 3 && nOut >= 4, "Unexpected EgoGeom graph IO counts");
    graphName = name;
    inputs.clear();
    outputs.clear();
    inputs.reserve(nIn);
    outputs.reserve(nOut);
    for (uint32_t i = 0; i < nIn; ++i) inputs.push_back(cloneTensorMeta(inT[i]));
    for (uint32_t i = 0; i < nOut; ++i) outputs.push_back(cloneTensorMeta(outT[i]));
}

void enableHtpBurstBestEffort(EgoGeomQnnEngine& eng) {
    if (!eng.qnn.deviceGetInfrastructure) return;
    QnnDevice_Infrastructure_t infra = nullptr;
    if (eng.qnn.deviceGetInfrastructure(&infra) != QNN_SUCCESS || !infra) return;
    const auto* htpInfra = reinterpret_cast<const QnnHtpDevice_Infrastructure_t*>(infra);
    if (htpInfra->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) return;
    eng.htpPerf = &htpInfra->perfInfra;
    if (!eng.htpPerf->createPowerConfigId || !eng.htpPerf->setPowerConfig) return;
    if (eng.htpPerf->createPowerConfigId(0, 0, &eng.powerConfigId) != QNN_SUCCESS) return;
    eng.hasPowerConfig = true;

    QnnHtpPerfInfrastructure_PowerConfig_t dcvs = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    dcvs.dcvsV3Config.contextId = eng.powerConfigId;
    dcvs.dcvsV3Config.setDcvsEnable = 1;
    dcvs.dcvsV3Config.dcvsEnable = 0; // disable DCVS -> lock burst corner
    dcvs.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    dcvs.dcvsV3Config.setSleepLatency = 1;
    dcvs.dcvsV3Config.sleepLatency = 40;
    dcvs.dcvsV3Config.setBusParams = 1;
    dcvs.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvs.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvs.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvs.dcvsV3Config.setCoreParams = 1;
    dcvs.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvs.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvs.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

    const QnnHtpPerfInfrastructure_PowerConfig_t* cfgs[] = {&dcvs, nullptr};
    auto rc = eng.htpPerf->setPowerConfig(eng.powerConfigId, cfgs);
    if (rc != QNN_SUCCESS) {
        LOGW("HTP burst power vote returned %llu (continuing with default clocks)",
             static_cast<unsigned long long>(rc));
    }
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_linnan_blindassist_vision_EgoGeomQnnRuntime_nativeCreate(
        JNIEnv* env, jobject, jstring bin_path, jstring dsp_dir) {
    try {
        const auto dsp = jstringToStd(env, dsp_dir);
        const char* prev = std::getenv("ADSP_LIBRARY_PATH");
        std::string search = dsp + ";" + (prev ? prev : "/vendor/lib/rfsa/adsp;/vendor/dsp;/system/lib/rfsa/adsp;/dsp");
        requireQnn(setenv("ADSP_LIBRARY_PATH", search.c_str(), 1) == 0, "Failed to set ADSP_LIBRARY_PATH");

        auto openLib = [](const char* name) -> void* {
            void* h = dlopen(name, RTLD_NOW | RTLD_NOLOAD);
            if (!h) h = dlopen(name, RTLD_NOW | RTLD_LOCAL);
            if (!h) {
                const char* err = dlerror();
                throw std::runtime_error(std::string("dlopen ") + name + " failed: " + (err ? err : "unknown"));
            }
            return h;
        };
        auto eng = std::make_unique<EgoGeomQnnEngine>();
        eng->backendLib = openLib("libQnnHtp.so");
        eng->sysLib = openLib("libQnnSystem.so");

        auto getProviders = reinterpret_cast<QnnInterfaceGetProvidersFn>(
                dlsym(eng->backendLib, "QnnInterface_getProviders"));
        requireQnn(getProviders != nullptr, "Missing QnnInterface_getProviders in libQnnHtp.so");
        const QnnInterface_t** providers = nullptr;
        uint32_t numProviders = 0;
        requireQnn(getProviders(&providers, &numProviders) == QNN_SUCCESS && numProviders > 0,
                   "QnnInterface_getProviders failed");
        eng->qnn = providers[0]->QNN_INTERFACE_VER_NAME;

        auto getSysProviders = reinterpret_cast<QnnSystemInterfaceGetProvidersFn>(
                dlsym(eng->sysLib, "QnnSystemInterface_getProviders"));
        requireQnn(getSysProviders != nullptr, "Missing QnnSystemInterface_getProviders in libQnnSystem.so");
        const QnnSystemInterface_t** sysProviders = nullptr;
        uint32_t numSysProviders = 0;
        requireQnn(getSysProviders(&sysProviders, &numSysProviders) == QNN_SUCCESS && numSysProviders > 0,
                   "QnnSystemInterface_getProviders failed");
        eng->qnnSys = sysProviders[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

        // Read precompiled context binary into memory
        const auto path = jstringToStd(env, bin_path);
        std::ifstream ifs(path, std::ios::binary | std::ios::ate);
        requireQnn(ifs.is_open(), "Unable to open context binary: " + path);
        const auto fileSize = static_cast<size_t>(ifs.tellg());
        requireQnn(fileSize > 0, "Empty context binary: " + path);
        ifs.seekg(0, std::ios::beg);
        std::vector<uint8_t> binData(fileSize);
        requireQnn(static_cast<bool>(ifs.read(reinterpret_cast<char*>(binData.data()), fileSize)),
                   "Failed reading context binary");

        // Parse graph/tensor metadata via QnnSystemContext
        QnnSystemContext_Handle_t sysCtx = nullptr;
        requireQnn(eng->qnnSys.systemContextCreate(&sysCtx) == QNN_SUCCESS && sysCtx != nullptr,
                   "systemContextCreate failed");
        const QnnSystemContext_BinaryInfo_t* binInfo = nullptr;
        Qnn_ContextBinarySize_t binInfoSize = 0;
        Qnn_ErrorHandle_t metaRc = QNN_COMMON_ERROR_GENERAL;
        if (eng->qnnSys.systemContextGetMetaData) {
            metaRc = eng->qnnSys.systemContextGetMetaData(sysCtx, binData.data(), binData.size(), &binInfo);
        }
        if (metaRc != QNN_SUCCESS && eng->qnnSys.systemContextGetBinaryInfo) {
            metaRc = eng->qnnSys.systemContextGetBinaryInfo(sysCtx, binData.data(), binData.size(), &binInfo, &binInfoSize);
        }
        try {
            requireQnn(metaRc == QNN_SUCCESS && binInfo != nullptr, "Failed reading context binary metadata");
            extractFirstGraphMeta(binInfo, eng->graphName, eng->inputs, eng->outputs);
        } catch (...) {
            eng->qnnSys.systemContextFree(sysCtx);
            throw;
        }
        eng->qnnSys.systemContextFree(sysCtx);
        eng->refreshTensorArrays();

        // Create QNN HTP backend & device
        requireQnn(eng->qnn.backendCreate(nullptr, nullptr, &eng->backend) == QNN_SUCCESS,
                   "QnnBackend_create failed");
        if (eng->qnn.deviceCreate) {
            auto devRc = eng->qnn.deviceCreate(nullptr, nullptr, &eng->device);
            requireQnn(devRc == QNN_SUCCESS || devRc == QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE,
                       "QnnDevice_create failed: " + std::to_string(devRc));
        }

        // Deserialize context binary directly on Hexagon v75
        auto ctxRc = eng->qnn.contextCreateFromBinary(
                eng->backend, eng->device, nullptr, binData.data(),
                static_cast<Qnn_ContextBinarySize_t>(binData.size()), &eng->context, nullptr);
        requireQnn(ctxRc == QNN_SUCCESS && eng->context != nullptr,
                   "QnnContext_createFromBinary failed: " + std::to_string(ctxRc));

        auto gRc = eng->qnn.graphRetrieve(eng->context, eng->graphName.c_str(), &eng->graph);
        requireQnn(gRc == QNN_SUCCESS && eng->graph != nullptr,
                   "QnnGraph_retrieve(" + eng->graphName + ") failed: " + std::to_string(gRc));

        enableHtpBurstBestEffort(*eng);
        LOGI("Loaded EgoGeom QNN context '%s' (%zu bytes, %zu inputs, %zu outputs)",
             eng->graphName.c_str(), fileSize, eng->inputs.size(), eng->outputs.size());
        return reinterpret_cast<jlong>(eng.release());
    } catch (const std::exception& e) {
        throwIllegalState(env, e);
        return 0;
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_linnan_blindassist_vision_EgoGeomQnnRuntime_nativeSupportsTof(JNIEnv*, jobject, jlong handle) {
    auto* eng = reinterpret_cast<EgoGeomQnnEngine*>(handle);
    return (eng && eng->inByName.count("tof_grid_s4") && eng->inByName.count("has_tof")) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_linnan_blindassist_vision_EgoGeomQnnRuntime_nativeSupportsGlass(JNIEnv*, jobject, jlong handle) {
    auto* eng = reinterpret_cast<EgoGeomQnnEngine*>(handle);
    return (eng && eng->outByName.count("glass_prob_s16")) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_linnan_blindassist_vision_EgoGeomQnnRuntime_nativeExecute(
        JNIEnv* env, jobject, jlong handle,
        jfloatArray rgb, jfloatArray grav_rays, jfloat host_focal,
        jfloatArray tof_grid_s4, jboolean has_tof,
        jfloatArray out_depth, jfloatArray out_dh, jfloatArray out_normal, jfloatArray out_ls,
        jfloatArray out_glass_prob) {
    try {
        auto* eng = reinterpret_cast<EgoGeomQnnEngine*>(handle);
        requireQnn(eng != nullptr, "EgoGeom QNN handle is null");

        auto copyIn = [&](const char* name, jfloatArray arr) {
            auto it = eng->inByName.find(name);
            requireQnn(it != eng->inByName.end(), std::string("Missing QNN input tensor: ") + name);
            auto& slot = eng->inputs[it->second];
            requireQnn(arr != nullptr && static_cast<size_t>(env->GetArrayLength(arr)) == slot.numElements,
                       std::string("Input size mismatch for ") + name);
            env->GetFloatArrayRegion(arr, 0, static_cast<jsize>(slot.numElements), slot.buf.data());
            requireQnn(!env->ExceptionCheck(), std::string("JNI copy failed for ") + name);
        };

        copyIn("rgb", rgb);
        copyIn("grav_rays", grav_rays);
        {
            auto it = eng->inByName.find("host_focal");
            requireQnn(it != eng->inByName.end(), "Missing QNN input tensor: host_focal");
            eng->inputs[it->second].buf[0] = static_cast<float>(host_focal);
        }
        if (eng->inByName.count("tof_grid_s4") && eng->inByName.count("has_tof")) {
            auto& tofSlot = eng->inputs[eng->inByName["tof_grid_s4"]];
            auto& htSlot = eng->inputs[eng->inByName["has_tof"]];
            if (has_tof == JNI_TRUE && tof_grid_s4 != nullptr) {
                requireQnn(static_cast<size_t>(env->GetArrayLength(tof_grid_s4)) == tofSlot.numElements,
                           "Input size mismatch for tof_grid_s4");
                env->GetFloatArrayRegion(tof_grid_s4, 0, static_cast<jsize>(tofSlot.numElements), tofSlot.buf.data());
                htSlot.buf[0] = 1.0f;
            } else {
                std::fill(tofSlot.buf.begin(), tofSlot.buf.end(), 0.0f);
                htSlot.buf[0] = 0.0f;
            }
        }

        auto rc = eng->qnn.graphExecute(
                eng->graph,
                eng->qnnInputs.data(), static_cast<uint32_t>(eng->qnnInputs.size()),
                eng->qnnOutputs.data(), static_cast<uint32_t>(eng->qnnOutputs.size()),
                nullptr, nullptr);
        requireQnn(rc == QNN_SUCCESS, "QnnGraph_execute failed: " + std::to_string(rc));

        auto copyOut = [&](const char* name, jfloatArray arr) {
            auto it = eng->outByName.find(name);
            requireQnn(it != eng->outByName.end(), std::string("Missing QNN output tensor: ") + name);
            const auto& slot = eng->outputs[it->second];
            requireQnn(arr != nullptr && static_cast<size_t>(env->GetArrayLength(arr)) == slot.numElements,
                       std::string("Output size mismatch for ") + name);
            env->SetFloatArrayRegion(arr, 0, static_cast<jsize>(slot.numElements), slot.buf.data());
            requireQnn(!env->ExceptionCheck(), std::string("JNI output copy failed for ") + name);
        };

        copyOut("depth_m", out_depth);
        copyOut("delta_h_m", out_dh);
        copyOut("normal_grav", out_normal);
        copyOut("log_sigma", out_ls);
        if (out_glass_prob != nullptr && eng->outByName.count("glass_prob_s16")) {
            copyOut("glass_prob_s16", out_glass_prob);
        }
    } catch (const std::exception& e) {
        throwIllegalState(env, e);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_linnan_blindassist_vision_EgoGeomQnnRuntime_nativeDestroy(JNIEnv*, jobject, jlong handle) {
    delete reinterpret_cast<EgoGeomQnnEngine*>(handle);
}
