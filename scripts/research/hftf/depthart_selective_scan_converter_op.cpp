#include "QnnOpPackage.h"
#include "QnnTypes.h"

#ifndef EXPORT_API
#if defined(_MSC_VER)
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__((visibility("default")))
#endif
#endif

extern "C" {

EXPORT_API Qnn_ErrorHandle_t SelectiveScanShapeInference(Qnn_OpConfig_t *op) {
  if (op == nullptr || op->v1.numOfInputs != 7 || op->v1.numOfOutputs != 1 ||
      op->v1.numOfParams != 2) {
    return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  }
  const Qnn_Tensor_t *input = &op->v1.inputTensors[0];
  Qnn_Tensor_t *output = &op->v1.outputTensors[0];
  if (input->v1.rank != 3 || output->v1.rank != input->v1.rank ||
      input->v1.dimensions == nullptr || output->v1.dimensions == nullptr) {
    return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  }
  for (uint32_t index = 0; index < input->v1.rank; ++index) {
    output->v1.dimensions[index] = input->v1.dimensions[index];
  }
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t (*SelectiveScanOutputInfoInferencePtr)(Qnn_OpConfig_t *) =
    &SelectiveScanShapeInference;

EXPORT_API Qnn_ErrorHandle_t SelectiveScanDataTypeInference(Qnn_OpConfig_t *op) {
  if (op == nullptr || op->v1.numOfInputs != 7 || op->v1.numOfOutputs != 1) {
    return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  }
  op->v1.outputTensors[0].v1.dataType = op->v1.inputTensors[0].v1.dataType;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t (*SelectiveScanDataTypeInferencePtr)(Qnn_OpConfig_t *) =
    &SelectiveScanDataTypeInference;

EXPORT_API Qnn_ErrorHandle_t DepthArtLayerNormShapeInference(Qnn_OpConfig_t *op) {
  if (op == nullptr || op->v1.numOfInputs != 3 || op->v1.numOfOutputs != 1 ||
      op->v1.numOfParams != 1) {
    return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  }
  const Qnn_Tensor_t *input = &op->v1.inputTensors[0];
  Qnn_Tensor_t *output = &op->v1.outputTensors[0];
  if (input->v1.rank != 3 || output->v1.rank != input->v1.rank ||
      input->v1.dimensions == nullptr || output->v1.dimensions == nullptr) {
    return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  }
  for (uint32_t index = 0; index < input->v1.rank; ++index) {
    output->v1.dimensions[index] = input->v1.dimensions[index];
  }
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t (*DepthArtLayerNormOutputInfoInferencePtr)(Qnn_OpConfig_t *) =
    &DepthArtLayerNormShapeInference;

EXPORT_API Qnn_ErrorHandle_t DepthArtLayerNormDataTypeInference(Qnn_OpConfig_t *op) {
  if (op == nullptr || op->v1.numOfInputs != 3 || op->v1.numOfOutputs != 1) {
    return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  }
  op->v1.outputTensors[0].v1.dataType = op->v1.inputTensors[0].v1.dataType;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t (*DepthArtLayerNormDataTypeInferencePtr)(Qnn_OpConfig_t *) =
    &DepthArtLayerNormDataTypeInference;

}  // extern "C"
