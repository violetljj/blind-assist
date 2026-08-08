// DepthART float32 last-axis LayerNorm HTP scalar reference kernel.

#include <cmath>

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

BEGIN_PKG_OP_DEFINITION(PKG_DepthArtLayerNorm);

static Qnn_Scalar_t sg_default_epsilon_scalar = {
    .dataType = Qnn_DataType_t::QNN_DATATYPE_FLOAT_32, .floatValue = 0.00001f};
static Qnn_Param_t sg_default_epsilon = {
    .paramType = QNN_PARAMTYPE_SCALAR, .scalarParam = sg_default_epsilon_scalar};

template <typename TensorType>
GraphStatus depthArtLayerNormReferenceImpl(TensorType &y,
                                           const TensorType &x,
                                           const TensorType &weight,
                                           const TensorType &bias,
                                           const Tensor &epsilon);

DEF_PACKAGE_OP_AND_COST_AND_FLAGS(
    (depthArtLayerNormReferenceImpl<Tensor>), "DepthArtLayerNorm", GLACIAL)

DEF_PACKAGE_PARAM_ORDER("DepthArtLayerNorm", "epsilon", false, &sg_default_epsilon)

template <typename TensorType>
GraphStatus depthArtLayerNormReferenceImpl(TensorType &y,
                                           const TensorType &x,
                                           const TensorType &weight,
                                           const TensorType &bias,
                                           const Tensor &epsilon) {
  const bool x_native = x.rank() == 3;
  const bool x_backfilled = x.rank() == 4 && x.dim(0) == 1;
  const bool y_native = y.rank() == 3;
  const bool y_backfilled = y.rank() == 4 && y.dim(0) == 1;
  const bool weight_native = weight.rank() == 1;
  const bool weight_backfilled = weight.rank() == 4 && weight.dim(0) == 1 &&
                                 weight.dim(1) == 1 && weight.dim(2) == 1;
  const bool bias_native = bias.rank() == 1;
  const bool bias_backfilled = bias.rank() == 4 && bias.dim(0) == 1 &&
                               bias.dim(1) == 1 && bias.dim(2) == 1;
  if ((!x_native && !x_backfilled) || (!y_native && !y_backfilled) ||
      (!weight_native && !weight_backfilled) ||
      (!bias_native && !bias_backfilled)) {
    return GraphStatus::ErrorFatal;
  }

  const size_t rows0 = x_native ? x.dim(0) : x.dim(1);
  const size_t rows1 = x_native ? x.dim(1) : x.dim(2);
  const size_t channels = x_native ? x.dim(2) : x.dim(3);
  const size_t weight_channels = weight_native ? weight.dim(0) : weight.dim(3);
  const size_t bias_channels = bias_native ? bias.dim(0) : bias.dim(3);
  if (channels == 0 || weight_channels != channels || bias_channels != channels) {
    return GraphStatus::ErrorFatal;
  }
  const float epsilon_value = epsilon.rank() == 1
                                  ? float(epsilon(0))
                                  : float(epsilon(0, 0, 0, 0));
  if (!(epsilon_value > 0.0f)) return GraphStatus::ErrorFatal;
  y.set_dims(x);

  for (size_t row0 = 0; row0 < rows0; ++row0) {
    for (size_t row1 = 0; row1 < rows1; ++row1) {
      float sum = 0.0f;
      float sum_compensation = 0.0f;
      for (size_t channel = 0; channel < channels; ++channel) {
        const float value = x_native ? float(x(row0, row1, channel))
                                     : float(x(0, row0, row1, channel));
        const float corrected = value - sum_compensation;
        const float next = sum + corrected;
        sum_compensation = (next - sum) - corrected;
        sum = next;
      }
      const float mean = sum / float(channels);
      float squared_sum = 0.0f;
      float squared_sum_compensation = 0.0f;
      for (size_t channel = 0; channel < channels; ++channel) {
        const float value = x_native ? float(x(row0, row1, channel))
                                     : float(x(0, row0, row1, channel));
        const float centered = value - mean;
        const float square = centered * centered;
        const float corrected = square - squared_sum_compensation;
        const float next = squared_sum + corrected;
        squared_sum_compensation = (next - squared_sum) - corrected;
        squared_sum = next;
      }
      const float inverse_std = 1.0f / std::sqrt(squared_sum / float(channels) + epsilon_value);
      for (size_t channel = 0; channel < channels; ++channel) {
        const float value = x_native ? float(x(row0, row1, channel))
                                     : float(x(0, row0, row1, channel));
        const float scale = weight_native ? float(weight(channel))
                                          : float(weight(0, 0, 0, channel));
        const float offset = bias_native ? float(bias(channel))
                                         : float(bias(0, 0, 0, channel));
        const float result = (value - mean) * inverse_std * scale + offset;
        if (y_native) {
          y(row0, row1, channel) = result;
        } else {
          y(0, row0, row1, channel) = result;
        }
      }
    }
  }
  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_DepthArtLayerNorm);
