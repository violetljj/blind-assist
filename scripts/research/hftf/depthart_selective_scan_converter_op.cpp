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

EXPORT_API Qnn_ErrorHandle_t DepthArtPatchConv2dShapeInference(Qnn_OpConfig_t *op) {
  if (op == nullptr || op->v1.numOfInputs != 2 || op->v1.numOfOutputs != 1 ||
      op->v1.numOfParams != 0) {
    return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  }
  const Qnn_Tensor_t *input = &op->v1.inputTensors[0];
  const Qnn_Tensor_t *weight = &op->v1.inputTensors[1];
  Qnn_Tensor_t *output = &op->v1.outputTensors[0];
  if (input->v1.rank != 4 || weight->v1.rank != 4 || output->v1.rank != 4 ||
      input->v1.dimensions == nullptr || weight->v1.dimensions == nullptr ||
      output->v1.dimensions == nullptr || input->v1.dimensions[1] != 3 ||
      weight->v1.dimensions[0] != 24 || weight->v1.dimensions[1] != 3 ||
      weight->v1.dimensions[2] != 3 || weight->v1.dimensions[3] != 3) {
    return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  }
  output->v1.dimensions[0] = input->v1.dimensions[0];
  output->v1.dimensions[1] = weight->v1.dimensions[0];
  output->v1.dimensions[2] = (input->v1.dimensions[2] + 1) / 2;
  output->v1.dimensions[3] = (input->v1.dimensions[3] + 1) / 2;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t (*DepthArtPatchConv2dOutputInfoInferencePtr)(Qnn_OpConfig_t *) =
    &DepthArtPatchConv2dShapeInference;

EXPORT_API Qnn_ErrorHandle_t DepthArtPatchConv2dDataTypeInference(Qnn_OpConfig_t *op) {
  if (op == nullptr || op->v1.numOfInputs != 2 || op->v1.numOfOutputs != 1) {
    return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  }
  op->v1.outputTensors[0].v1.dataType = op->v1.inputTensors[0].v1.dataType;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t (*DepthArtPatchConv2dDataTypeInferencePtr)(Qnn_OpConfig_t *) =
    &DepthArtPatchConv2dDataTypeInference;

EXPORT_API Qnn_ErrorHandle_t DepthArtBatchNorm2dShapeInference(Qnn_OpConfig_t *op) {
  if (op == nullptr || op->v1.numOfInputs != 5 || op->v1.numOfOutputs != 1 ||
      op->v1.numOfParams != 1) {
    return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  }
  const Qnn_Tensor_t *input = &op->v1.inputTensors[0];
  Qnn_Tensor_t *output = &op->v1.outputTensors[0];
  if (input->v1.rank != 4 || output->v1.rank != 4 ||
      input->v1.dimensions == nullptr || output->v1.dimensions == nullptr) {
    return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  }
  for (uint32_t index = 0; index < input->v1.rank; ++index) {
    output->v1.dimensions[index] = input->v1.dimensions[index];
  }
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t (*DepthArtBatchNorm2dOutputInfoInferencePtr)(Qnn_OpConfig_t *) =
    &DepthArtBatchNorm2dShapeInference;

EXPORT_API Qnn_ErrorHandle_t DepthArtBatchNorm2dDataTypeInference(Qnn_OpConfig_t *op) {
  if (op == nullptr || op->v1.numOfInputs != 5 || op->v1.numOfOutputs != 1) {
    return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  }
  op->v1.outputTensors[0].v1.dataType = op->v1.inputTensors[0].v1.dataType;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t (*DepthArtBatchNorm2dDataTypeInferencePtr)(Qnn_OpConfig_t *) =
    &DepthArtBatchNorm2dDataTypeInference;

EXPORT_API Qnn_ErrorHandle_t DepthArtGeluShapeInference(Qnn_OpConfig_t *op) {
  if (op == nullptr || op->v1.numOfInputs != 1 || op->v1.numOfOutputs != 1 ||
      op->v1.numOfParams != 0) {
    return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  }
  const Qnn_Tensor_t *input = &op->v1.inputTensors[0];
  Qnn_Tensor_t *output = &op->v1.outputTensors[0];
  if (input->v1.rank == 0 || input->v1.rank > 4 ||
      output->v1.rank != input->v1.rank || input->v1.dimensions == nullptr ||
      output->v1.dimensions == nullptr) {
    return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  }
  for (uint32_t index = 0; index < input->v1.rank; ++index) {
    output->v1.dimensions[index] = input->v1.dimensions[index];
  }
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t (*DepthArtGeluOutputInfoInferencePtr)(Qnn_OpConfig_t *) =
    &DepthArtGeluShapeInference;

EXPORT_API Qnn_ErrorHandle_t DepthArtGeluDataTypeInference(Qnn_OpConfig_t *op) {
  if (op == nullptr || op->v1.numOfInputs != 1 || op->v1.numOfOutputs != 1) {
    return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  }
  op->v1.outputTensors[0].v1.dataType = op->v1.inputTensors[0].v1.dataType;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t (*DepthArtGeluDataTypeInferencePtr)(Qnn_OpConfig_t *) =
    &DepthArtGeluDataTypeInference;

}  // extern "C"
