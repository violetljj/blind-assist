#include <jni.h>
#include <android/log.h>
#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>

#if DAV2_QNN_AVAILABLE
#include "QnnInterface.h"
#include "HTP/QnnHtpDevice.h"
#include "System/QnnSystemInterface.h"

namespace {
constexpr const char* kTag = "Dav2QnnCachedR0";

using GetQnnProviders = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
using GetQnnSystemProviders = Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
T symbol(void* library, const char* name) {
    auto result = reinterpret_cast<T>(dlsym(library, name));
    if (result == nullptr) {
        const char* error = dlerror();
        throw std::runtime_error(
                std::string("missing symbol ") + name + ": " +
                (error != nullptr ? error : "unknown dynamic-loader error"));
    }
    return result;
}

struct GraphMetadata {
    const char* name = nullptr;
    uint32_t input_count = 0;
    Qnn_Tensor_t* inputs = nullptr;
    uint32_t output_count = 0;
    Qnn_Tensor_t* outputs = nullptr;
};

GraphMetadata graph_metadata(const QnnSystemContext_GraphInfo_t& graph) {
    switch (graph.version) {
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1:
            return {graph.graphInfoV1.graphName, graph.graphInfoV1.numGraphInputs,
                    graph.graphInfoV1.graphInputs, graph.graphInfoV1.numGraphOutputs,
                    graph.graphInfoV1.graphOutputs};
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2:
            return {graph.graphInfoV2.graphName, graph.graphInfoV2.numGraphInputs,
                    graph.graphInfoV2.graphInputs, graph.graphInfoV2.numGraphOutputs,
                    graph.graphInfoV2.graphOutputs};
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3:
            return {graph.graphInfoV3.graphName, graph.graphInfoV3.numGraphInputs,
                    graph.graphInfoV3.graphInputs, graph.graphInfoV3.numGraphOutputs,
                    graph.graphInfoV3.graphOutputs};
        default:
            throw std::runtime_error("unsupported graph metadata version");
    }
}

struct BinaryMetadata {
    uint32_t graph_count = 0;
    QnnSystemContext_GraphInfo_t* graphs = nullptr;
};

BinaryMetadata binary_metadata(const QnnSystemContext_BinaryInfo_t& binary) {
    switch (binary.version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
            return {binary.contextBinaryInfoV1.numGraphs, binary.contextBinaryInfoV1.graphs};
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
            return {binary.contextBinaryInfoV2.numGraphs, binary.contextBinaryInfoV2.graphs};
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
            return {binary.contextBinaryInfoV3.numGraphs, binary.contextBinaryInfoV3.graphs};
        default:
            throw std::runtime_error("unsupported binary metadata version");
    }
}

uint32_t element_size(Qnn_DataType_t type) {
    switch (type) {
        case QNN_DATATYPE_FLOAT_16: return 2;
        case QNN_DATATYPE_FLOAT_32: return 4;
        default: throw std::runtime_error("only FP16/FP32 client tensors are supported");
    }
}

const char* type_name(Qnn_DataType_t type) {
    switch (type) {
        case QNN_DATATYPE_FLOAT_16: return "FLOAT_16";
        case QNN_DATATYPE_FLOAT_32: return "FLOAT_32";
        default: return "UNSUPPORTED";
    }
}

uint64_t element_count(const Qnn_Tensor_t& tensor) {
    uint64_t count = 1;
    for (uint32_t index = 0; index < tensor.v1.rank; ++index) count *= tensor.v1.dimensions[index];
    return count;
}

std::string dims_json(const Qnn_Tensor_t& tensor) {
    std::ostringstream output;
    output << '[';
    for (uint32_t index = 0; index < tensor.v1.rank; ++index) {
        if (index) output << ',';
        output << tensor.v1.dimensions[index];
    }
    output << ']';
    return output.str();
}

struct CachedContext {
    void* backend_library = nullptr;
    void* system_library = nullptr;
    QNN_INTERFACE_VER_TYPE qnn = QNN_INTERFACE_VER_TYPE_INIT;
    QNN_SYSTEM_INTERFACE_VER_TYPE system = QNN_SYSTEM_INTERFACE_VER_TYPE_INIT;
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    QnnHtpDevice_PerfInfrastructure_t* perf_infrastructure = nullptr;
    uint32_t power_config_id = 0;
    bool power_config_created = false;
    Qnn_ErrorHandle_t rpc_polling_error = QNN_SUCCESS;
    Qnn_ContextHandle_t context = nullptr;
    Qnn_GraphHandle_t graph = nullptr;
    QnnSystemDlc_Handle_t dlc = nullptr;
    QnnSystemDlc_RecordHandle_t* records = nullptr;
    uint32_t record_count = 0;
    QnnSystemContext_Handle_t system_context = nullptr;
    Qnn_Tensor_t input = QNN_TENSOR_INIT;
    Qnn_Tensor_t output = QNN_TENSOR_INIT;
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    std::string graph_name;

    CachedContext(const char* dlc_path, const char* backend_path, const char* system_path) {
        const std::string backend_path_string(backend_path);
        const auto separator = backend_path_string.find_last_of('/');
        const std::string native_dir = separator == std::string::npos
            ? "."
            : backend_path_string.substr(0, separator);
        const std::string adsp_path = native_dir + ";/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp";
        setenv("ADSP_LIBRARY_PATH", adsp_path.c_str(), 1);
        backend_library = dlopen(backend_path, RTLD_NOW | RTLD_GLOBAL);
        if (backend_library == nullptr) {
            const char* error = dlerror();
            throw std::runtime_error(
                    std::string("dlopen backend failed: ") +
                    (error != nullptr ? error : "unknown dynamic-loader error"));
        }
        system_library = dlopen(system_path, RTLD_NOW | RTLD_LOCAL);
        if (system_library == nullptr) {
            const char* error = dlerror();
            throw std::runtime_error(
                    std::string("dlopen system failed: ") +
                    (error != nullptr ? error : "unknown dynamic-loader error"));
        }

        const QnnInterface_t** qnn_providers = nullptr;
        uint32_t qnn_provider_count = 0;
        auto get_qnn = symbol<GetQnnProviders>(backend_library, "QnnInterface_getProviders");
        check(get_qnn(&qnn_providers, &qnn_provider_count) == QNN_SUCCESS,
              "QnnInterface_getProviders failed");
        bool qnn_found = false;
        for (uint32_t index = 0; index < qnn_provider_count; ++index) {
            const auto* provider = qnn_providers[index];
            if (provider->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
                provider->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
                qnn = provider->QNN_INTERFACE_VER_NAME;
                qnn_found = true;
                break;
            }
        }
        check(qnn_found, "compatible QNN backend interface unavailable");

        const QnnSystemInterface_t** system_providers = nullptr;
        uint32_t system_provider_count = 0;
        auto get_system = symbol<GetQnnSystemProviders>(system_library, "QnnSystemInterface_getProviders");
        check(get_system(&system_providers, &system_provider_count) == QNN_SUCCESS,
              "QnnSystemInterface_getProviders failed");
        bool system_found = false;
        for (uint32_t index = 0; index < system_provider_count; ++index) {
            const auto* provider = system_providers[index];
            if (provider->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR &&
                provider->systemApiVersion.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
                system = provider->QNN_SYSTEM_INTERFACE_VER_NAME;
                system_found = true;
                break;
            }
        }
        check(system_found, "compatible QNN system interface unavailable");
        check(qnn.backendCreate(nullptr, nullptr, &backend) == QNN_BACKEND_NO_ERROR,
              "backendCreate failed");
        check(qnn.deviceCreate != nullptr &&
                  qnn.deviceCreate(nullptr, nullptr, &device) == QNN_DEVICE_NO_ERROR,
              "deviceCreate failed");
        QnnDevice_Infrastructure_t device_infrastructure = nullptr;
        check(qnn.deviceGetInfrastructure != nullptr &&
                  qnn.deviceGetInfrastructure(&device_infrastructure) == QNN_SUCCESS &&
                  device_infrastructure != nullptr,
              "HTP performance infrastructure unavailable");
        auto* htp_infrastructure =
                reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(device_infrastructure);
        check(htp_infrastructure->infraType == QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF,
              "unexpected HTP infrastructure type");
        perf_infrastructure = &htp_infrastructure->perfInfra;
        check(perf_infrastructure->createPowerConfigId != nullptr &&
                  perf_infrastructure->setPowerConfig != nullptr &&
                  perf_infrastructure->destroyPowerConfigId != nullptr,
              "incomplete HTP performance infrastructure");
        check(perf_infrastructure->createPowerConfigId(0, 0, &power_config_id) == QNN_SUCCESS,
              "createPowerConfigId failed");
        power_config_created = true;

        QnnHtpPerfInfrastructure_PowerConfig_t dcvs =
                QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
        dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        dcvs.dcvsV3Config.contextId = power_config_id;
        dcvs.dcvsV3Config.setDcvsEnable = 1;
        dcvs.dcvsV3Config.dcvsEnable = 0;
        dcvs.dcvsV3Config.powerMode =
                QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
        dcvs.dcvsV3Config.setSleepLatency = 1;
        dcvs.dcvsV3Config.sleepLatency = 100;
        dcvs.dcvsV3Config.setSleepDisable = 0;
        dcvs.dcvsV3Config.sleepDisable = 0;
        dcvs.dcvsV3Config.setBusParams = 1;
        dcvs.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO;
        dcvs.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
        dcvs.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO;
        dcvs.dcvsV3Config.setCoreParams = 1;
        dcvs.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO;
        dcvs.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
        dcvs.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO;

        QnnHtpPerfInfrastructure_PowerConfig_t polling =
                QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
        polling.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
        polling.rpcPollingTimeConfig =
                QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_MAX_RPC_POLLING_TIME;
        const QnnHtpPerfInfrastructure_PowerConfig_t* dcvs_configs[] = {&dcvs, nullptr};
        const auto dcvs_error =
                perf_infrastructure->setPowerConfig(power_config_id, dcvs_configs);
        check(dcvs_error == QNN_SUCCESS,
              std::string("sustained-high-performance DCVS vote failed: ") +
                  std::to_string(dcvs_error));
        const QnnHtpPerfInfrastructure_PowerConfig_t* polling_configs[] = {&polling, nullptr};
        rpc_polling_error =
                perf_infrastructure->setPowerConfig(power_config_id, polling_configs);
        check(system.systemDlcCreateFromFile != nullptr, "system DLC API unavailable");
        check(system.systemDlcCreateFromFile(nullptr, dlc_path, &dlc) == QNN_SUCCESS,
              "systemDlcCreateFromFile failed");
        check(system.systemDlcGetRecordsByType(
                  dlc,
                  QNN_SYSTEM_DLC_RECORD_PREFIX_HTP_CACHE_RECORD,
                  1,
                  &records,
                  &record_count) == QNN_SUCCESS && record_count > 0,
              "no compatible HTP cached context record");
        const uint8_t* binary = nullptr;
        uint64_t binary_size = 0;
        check(system.systemDlcReadRecordDataMemoryMapped(records[0], &binary, &binary_size) == QNN_SUCCESS &&
                  binary != nullptr && binary_size > 0,
              "unable to map cached context record");

        check(system.systemContextCreate(&system_context) == QNN_SUCCESS,
              "systemContextCreate failed");
        const QnnSystemContext_BinaryInfo_t* binary_info = nullptr;
        Qnn_ContextBinarySize_t metadata_size = 0;
        check(system.systemContextGetBinaryInfo(
                  system_context,
                  const_cast<uint8_t*>(binary),
                  binary_size,
                  &binary_info,
                  &metadata_size) == QNN_SUCCESS && binary_info != nullptr,
              "systemContextGetBinaryInfo failed");
        const auto binary_view = binary_metadata(*binary_info);
        check(binary_view.graph_count == 1 && binary_view.graphs != nullptr,
              "cached context must contain exactly one graph");
        const auto graph_view = graph_metadata(binary_view.graphs[0]);
        check(graph_view.name != nullptr && graph_view.input_count == 1 && graph_view.output_count == 1,
              "cached graph must have one input and one output");
        graph_name = graph_view.name;
        input = graph_view.inputs[0];
        output = graph_view.outputs[0];
        input_bytes = element_count(input) * element_size(input.v1.dataType);
        output_bytes = element_count(output) * element_size(output.v1.dataType);

        check(qnn.contextCreateFromBinary(
                  backend,
                  device,
                  nullptr,
                  const_cast<uint8_t*>(binary),
                  binary_size,
                  &context,
                  nullptr) == QNN_CONTEXT_NO_ERROR,
              "contextCreateFromBinary failed");
        check(qnn.graphRetrieve(context, graph_name.c_str(), &graph) == QNN_GRAPH_NO_ERROR,
              "graphRetrieve failed");
    }

    ~CachedContext() {
        if (context && qnn.contextFree) qnn.contextFree(context, nullptr);
        if (system_context && system.systemContextFree) system.systemContextFree(system_context);
        if (records && system.systemDlcFreeRecord) {
            for (uint32_t index = 0; index < record_count; ++index) system.systemDlcFreeRecord(records[index]);
        }
        if (dlc && system.systemDlcFree) system.systemDlcFree(dlc);
        if (power_config_created && perf_infrastructure &&
            perf_infrastructure->destroyPowerConfigId) {
            perf_infrastructure->destroyPowerConfigId(power_config_id);
        }
        if (device && qnn.deviceFree) qnn.deviceFree(device);
        if (backend && qnn.backendFree) qnn.backendFree(backend);
        if (system_library) dlclose(system_library);
        if (backend_library) dlclose(backend_library);
    }

    void execute(void* input_data, uint64_t input_capacity, void* output_data, uint64_t output_capacity) {
        check(input_capacity >= input_bytes, "input direct buffer is too small");
        check(output_capacity >= output_bytes, "output direct buffer is too small");
        input.v1.memType = QNN_TENSORMEMTYPE_RAW;
        input.v1.clientBuf = {input_data, static_cast<uint32_t>(input_bytes)};
        output.v1.memType = QNN_TENSORMEMTYPE_RAW;
        output.v1.clientBuf = {output_data, static_cast<uint32_t>(output_bytes)};
        check(qnn.graphExecute(graph, &input, 1, &output, 1, nullptr, nullptr) == QNN_GRAPH_NO_ERROR,
              "graphExecute failed");
    }

    std::string metadata_json() const {
        std::ostringstream value;
        value << "{\"graph_name\":\"" << graph_name << "\","
              << "\"performance_profile\":\"sustained_high_performance\","
              << "\"rpc_polling_error\":" << rpc_polling_error << ','
              << "\"input_name\":\"" << (input.v1.name ? input.v1.name : "") << "\","
              << "\"input_type\":\"" << type_name(input.v1.dataType) << "\","
              << "\"input_dims\":" << dims_json(input) << ','
              << "\"input_bytes\":" << input_bytes << ','
              << "\"output_name\":\"" << (output.v1.name ? output.v1.name : "") << "\","
              << "\"output_type\":\"" << type_name(output.v1.dataType) << "\","
              << "\"output_dims\":" << dims_json(output) << ','
              << "\"output_bytes\":" << output_bytes << '}';
        return value.str();
    }
};

void throw_java(JNIEnv* env, const std::exception& error) {
    __android_log_print(ANDROID_LOG_ERROR, kTag, "%s", error.what());
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error.what());
}
}  // namespace
#endif

extern "C" JNIEXPORT jlong JNICALL
Java_com_linnan_blindassist_hftf_Dav2QnnCachedContext_nativeCreate(
        JNIEnv* env, jobject, jstring dlc_path, jstring backend_path, jstring system_path) {
#if DAV2_QNN_AVAILABLE
    const char* dlc = env->GetStringUTFChars(dlc_path, nullptr);
    const char* backend = env->GetStringUTFChars(backend_path, nullptr);
    const char* system = env->GetStringUTFChars(system_path, nullptr);
    try {
        auto* context = new CachedContext(dlc, backend, system);
        env->ReleaseStringUTFChars(dlc_path, dlc);
        env->ReleaseStringUTFChars(backend_path, backend);
        env->ReleaseStringUTFChars(system_path, system);
        return reinterpret_cast<jlong>(context);
    } catch (const std::exception& error) {
        env->ReleaseStringUTFChars(dlc_path, dlc);
        env->ReleaseStringUTFChars(backend_path, backend);
        env->ReleaseStringUTFChars(system_path, system);
        throw_java(env, error);
        return 0;
    }
#else
    env->ThrowNew(env->FindClass("java/lang/UnsupportedOperationException"), "QAIRT headers unavailable at build time");
    return 0;
#endif
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_linnan_blindassist_hftf_Dav2QnnCachedContext_nativeMetadata(JNIEnv* env, jobject, jlong handle) {
#if DAV2_QNN_AVAILABLE
    try {
        auto* context = reinterpret_cast<CachedContext*>(handle);
        check(context != nullptr, "cached context is closed");
        return env->NewStringUTF(context->metadata_json().c_str());
    } catch (const std::exception& error) {
        throw_java(env, error);
        return nullptr;
    }
#else
    return env->NewStringUTF("{}");
#endif
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_linnan_blindassist_hftf_Dav2QnnCachedContext_nativeExecute(
        JNIEnv* env, jobject, jlong handle, jobject input, jobject output,
        jboolean compute_input_hash) {
#if DAV2_QNN_AVAILABLE
    try {
        auto* context = reinterpret_cast<CachedContext*>(handle);
        check(context != nullptr, "cached context is closed");
        void* input_data = env->GetDirectBufferAddress(input);
        void* output_data = env->GetDirectBufferAddress(output);
        check(input_data != nullptr && output_data != nullptr, "QNN tensors must be direct buffers");
        const uint64_t input_bytes = context->input_bytes;
        uint64_t hash = 0;
        if (compute_input_hash) {
            const auto* bytes = static_cast<const uint8_t*>(input_data);
            hash = 14695981039346656037ull;
            for (uint64_t index = 0; index < input_bytes; ++index) {
                hash ^= bytes[index];
                hash *= 1099511628211ull;
            }
        }
        context->execute(
            input_data,
            env->GetDirectBufferCapacity(input),
            output_data,
            env->GetDirectBufferCapacity(output));
        return static_cast<jlong>(hash);
    } catch (const std::exception& error) {
        throw_java(env, error);
        return 0;
    }
#else
    env->ThrowNew(env->FindClass("java/lang/UnsupportedOperationException"), "QAIRT headers unavailable at build time");
    return 0;
#endif
}

extern "C" JNIEXPORT void JNICALL
Java_com_linnan_blindassist_hftf_Dav2QnnCachedContext_nativeDestroy(JNIEnv*, jobject, jlong handle) {
#if DAV2_QNN_AVAILABLE
    delete reinterpret_cast<CachedContext*>(handle);
#endif
}
