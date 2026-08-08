// DepthART SelectiveScan HTP scalar reference kernel.
//
// This kernel is intentionally correctness-first. It uses constant stack
// storage and no heap allocation, but it is not an HVX-optimized performance
// implementation. Build/runtime evidence is required before claiming HTP
// execution.

#include <array>
#include <cmath>

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

BEGIN_PKG_OP_DEFINITION(PKG_SelectiveScan);

static Qnn_Scalar_t sg_default_delta_softplus_scalar = {
    .dataType = Qnn_DataType_t::QNN_DATATYPE_BOOL_8, .bool8Value = 1};
static Qnn_Param_t sg_default_delta_softplus = {
    .paramType = QNN_PARAMTYPE_SCALAR, .scalarParam = sg_default_delta_softplus_scalar};
static Qnn_Scalar_t sg_default_out_float_scalar = {
    .dataType = Qnn_DataType_t::QNN_DATATYPE_BOOL_8, .bool8Value = 0};
static Qnn_Param_t sg_default_out_float = {
    .paramType = QNN_PARAMTYPE_SCALAR, .scalarParam = sg_default_out_float_scalar};

template <typename TensorType>
GraphStatus selectivescanReferenceImpl(TensorType &y,
                                       const TensorType &u,
                                       const TensorType &delta,
                                       const TensorType &A,
                                       const TensorType &B,
                                       const TensorType &C,
                                       const TensorType &D,
                                       const TensorType &delta_bias,
                                       const QuantUint16Tensor &delta_softplus,
                                       const QuantUint16Tensor &out_float);

DEF_PACKAGE_OP_AND_COST_AND_FLAGS(
    (selectivescanReferenceImpl<Tensor>), "SelectiveScan", GLACIAL)

DEF_TENSOR_PROPERTIES(Op("SelectiveScan", "u", "delta", "A", "B", "C", "D", "delta_bias"),
                      Outputs("y"),
                      Flat("*"),
                      MainMemory("*"))

DEF_PACKAGE_PARAM_ORDER("SelectiveScan",
                        "delta_softplus",
                        false,
                        &sg_default_delta_softplus,
                        "out_float",
                        false,
                        &sg_default_out_float)

static inline float stableSoftplus(float value) {
  if (value > 20.0f) return value;
  if (value < -20.0f) return std::exp(value);
  return std::log1p(std::exp(value));
}

template <typename TensorType>
GraphStatus selectivescanReferenceImpl(TensorType &y,
                                       const TensorType &u,
                                       const TensorType &delta,
                                       const TensorType &A,
                                       const TensorType &B,
                                       const TensorType &C,
                                       const TensorType &D,
                                       const TensorType &delta_bias,
                                       const QuantUint16Tensor &delta_softplus,
                                       const QuantUint16Tensor &out_float) {
  constexpr size_t kFrozenGroups = 4;
  constexpr size_t kFrozenStateDim = 8;

  if (u.rank() != 3 || delta.rank() != 3 || A.rank() != 2 || B.rank() != 4 ||
      C.rank() != 4 || D.rank() != 1 || delta_bias.rank() != 1 || y.rank() != 3) {
    return GraphStatus::ErrorFatal;
  }
  const size_t batch = u.dim(0);
  const size_t channels = u.dim(1);
  const size_t length = u.dim(2);
  if (delta.dim(0) != batch || delta.dim(1) != channels || delta.dim(2) != length ||
      A.dim(0) != channels || A.dim(1) != kFrozenStateDim ||
      B.dim(0) != batch || B.dim(1) != kFrozenGroups ||
      B.dim(2) != kFrozenStateDim || B.dim(3) != length ||
      C.dim(0) != batch || C.dim(1) != kFrozenGroups ||
      C.dim(2) != kFrozenStateDim || C.dim(3) != length ||
      D.dim(0) != channels || delta_bias.dim(0) != channels ||
      channels % kFrozenGroups != 0) {
    return GraphStatus::ErrorFatal;
  }
  if (float(delta_softplus(0, 0, 0, 0)) < 0.5f ||
      float(out_float(0, 0, 0, 0)) >= 0.5f) {
    return GraphStatus::ErrorFatal;
  }
  y.set_dims(u);

  const size_t channels_per_group = channels / kFrozenGroups;
  for (size_t batch_index = 0; batch_index < batch; ++batch_index) {
    for (size_t channel = 0; channel < channels; ++channel) {
      const size_t group = channel / channels_per_group;
      std::array<float, kFrozenStateDim> state{};
      for (size_t step = 0; step < length; ++step) {
        const float input = u(batch_index, channel, step);
        const float dt = stableSoftplus(float(delta(batch_index, channel, step)) +
                                        float(delta_bias(channel)));
        float value = 0.0f;
        for (size_t state_index = 0; state_index < kFrozenStateDim; ++state_index) {
          const float transition = std::exp(dt * float(A(channel, state_index)));
          state[state_index] =
              transition * state[state_index] +
              dt * float(B(batch_index, group, state_index, step)) * input;
          value += state[state_index] * float(C(batch_index, group, state_index, step));
        }
        y(batch_index, channel, step) = value + float(D(channel)) * input;
      }
    }
  }
  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_SelectiveScan);
