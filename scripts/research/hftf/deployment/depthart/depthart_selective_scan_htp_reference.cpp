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
                                       const Tensor &delta_softplus,
                                       const Tensor &out_float);

DEF_PACKAGE_OP_AND_COST_AND_FLAGS(
    (selectivescanReferenceImpl<Tensor>), "SelectiveScan", GLACIAL)

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
                                       const Tensor &delta_softplus,
                                       const Tensor &out_float) {
  constexpr size_t kFrozenGroups = 4;
  constexpr size_t kFrozenStateDim = 8;

  const bool u_is_native_rank = u.rank() == 3;
  const bool u_is_backfilled = u.rank() == 4 && u.dim(0) == 1;
  const bool delta_is_native_rank = delta.rank() == 3;
  const bool delta_is_backfilled = delta.rank() == 4 && delta.dim(0) == 1;
  const bool a_is_native_rank = A.rank() == 2;
  const bool a_is_backfilled = A.rank() == 4 && A.dim(0) == 1 && A.dim(1) == 1;
  const bool d_is_native_rank = D.rank() == 1;
  const bool d_is_backfilled = D.rank() == 4 && D.dim(0) == 1 && D.dim(1) == 1 && D.dim(2) == 1;
  const bool bias_is_native_rank = delta_bias.rank() == 1;
  const bool bias_is_backfilled = delta_bias.rank() == 4 && delta_bias.dim(0) == 1 &&
                                  delta_bias.dim(1) == 1 && delta_bias.dim(2) == 1;
  const bool y_is_native_rank = y.rank() == 3;
  const bool y_is_backfilled = y.rank() == 4 && y.dim(0) == 1;
  if ((!u_is_native_rank && !u_is_backfilled) ||
      (!delta_is_native_rank && !delta_is_backfilled) ||
      (!a_is_native_rank && !a_is_backfilled) || B.rank() != 4 || C.rank() != 4 ||
      (!d_is_native_rank && !d_is_backfilled) ||
      (!bias_is_native_rank && !bias_is_backfilled) ||
      (!y_is_native_rank && !y_is_backfilled)) {
    return GraphStatus::ErrorFatal;
  }
  const size_t batch = u_is_native_rank ? u.dim(0) : u.dim(1);
  const size_t channels = u_is_native_rank ? u.dim(1) : u.dim(2);
  const size_t length = u_is_native_rank ? u.dim(2) : u.dim(3);
  const size_t delta_batch = delta_is_native_rank ? delta.dim(0) : delta.dim(1);
  const size_t delta_channels = delta_is_native_rank ? delta.dim(1) : delta.dim(2);
  const size_t delta_length = delta_is_native_rank ? delta.dim(2) : delta.dim(3);
  const size_t a_channels = a_is_native_rank ? A.dim(0) : A.dim(2);
  const size_t a_state_dim = a_is_native_rank ? A.dim(1) : A.dim(3);
  const size_t d_channels = d_is_native_rank ? D.dim(0) : D.dim(3);
  const size_t bias_channels = bias_is_native_rank ? delta_bias.dim(0) : delta_bias.dim(3);
  if (delta_batch != batch || delta_channels != channels || delta_length != length ||
      a_channels != channels || a_state_dim != kFrozenStateDim ||
      B.dim(0) != batch || B.dim(1) != kFrozenGroups ||
      B.dim(2) != kFrozenStateDim || B.dim(3) != length ||
      C.dim(0) != batch || C.dim(1) != kFrozenGroups ||
      C.dim(2) != kFrozenStateDim || C.dim(3) != length ||
      d_channels != channels || bias_channels != channels ||
      channels % kFrozenGroups != 0) {
    return GraphStatus::ErrorFatal;
  }
  const float delta_softplus_value = delta_softplus.rank() == 1
                                         ? float(delta_softplus(0))
                                         : float(delta_softplus(0, 0, 0, 0));
  const float out_float_value = out_float.rank() == 1
                                    ? float(out_float(0))
                                    : float(out_float(0, 0, 0, 0));
  if (delta_softplus_value < 0.5f || out_float_value >= 0.5f) {
    return GraphStatus::ErrorFatal;
  }
  y.set_dims(u);

  const size_t channels_per_group = channels / kFrozenGroups;
  for (size_t batch_index = 0; batch_index < batch; ++batch_index) {
    for (size_t channel = 0; channel < channels; ++channel) {
      const size_t group = channel / channels_per_group;
      std::array<float, kFrozenStateDim> state{};
      for (size_t step = 0; step < length; ++step) {
        const float input = u_is_native_rank ? float(u(batch_index, channel, step))
                                             : float(u(0, batch_index, channel, step));
        const float delta_value = delta_is_native_rank
                                      ? float(delta(batch_index, channel, step))
                                      : float(delta(0, batch_index, channel, step));
        const float bias_value = bias_is_native_rank ? float(delta_bias(channel))
                                                     : float(delta_bias(0, 0, 0, channel));
        const float dt = stableSoftplus(delta_value + bias_value);
        float value = 0.0f;
        for (size_t state_index = 0; state_index < kFrozenStateDim; ++state_index) {
          const float a_value = a_is_native_rank ? float(A(channel, state_index))
                                                 : float(A(0, 0, channel, state_index));
          const float transition = std::exp(dt * a_value);
          state[state_index] =
              transition * state[state_index] +
              dt * float(B(batch_index, group, state_index, step)) * input;
          value += state[state_index] * float(C(batch_index, group, state_index, step));
        }
        const float d_value = d_is_native_rank ? float(D(channel)) : float(D(0, 0, 0, channel));
        if (y_is_native_rank) {
          y(batch_index, channel, step) = value + d_value * input;
        } else {
          y(0, batch_index, channel, step) = value + d_value * input;
        }
      }
    }
  }
  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_SelectiveScan);
