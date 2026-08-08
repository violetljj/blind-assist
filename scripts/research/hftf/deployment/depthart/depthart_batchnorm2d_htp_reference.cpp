// DepthART float32 inference BatchNormalization HTP scalar reference kernel.

#include <cmath>
#include <cstddef>

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

BEGIN_PKG_OP_DEFINITION(PKG_DepthArtBatchNorm2d);

static Qnn_Scalar_t sg_default_epsilon_scalar = {
    .dataType = QNN_DATATYPE_FLOAT_32, .floatValue = 0.00001f};
static Qnn_Param_t sg_default_epsilon = {
    .paramType = QNN_PARAMTYPE_SCALAR, .scalarParam = sg_default_epsilon_scalar};

template <typename TensorType>
GraphStatus depthArtBatchNorm2dReferenceImpl(TensorType &y,
                                             const TensorType &x,
                                             const TensorType &scale,
                                             const TensorType &bias,
                                             const TensorType &mean,
                                             const TensorType &variance,
                                             const Tensor &epsilon);

DEF_PACKAGE_OP_AND_COST_AND_FLAGS(
    (depthArtBatchNorm2dReferenceImpl<Tensor>), "DepthArtBatchNorm2d", GLACIAL)

DEF_PACKAGE_PARAM_ORDER("DepthArtBatchNorm2d", "epsilon", false, &sg_default_epsilon)

template <typename TensorType>
GraphStatus depthArtBatchNorm2dReferenceImpl(TensorType &y,
                                             const TensorType &x,
                                             const TensorType &scale,
                                             const TensorType &bias,
                                             const TensorType &mean,
                                             const TensorType &variance,
                                             const Tensor &epsilon) {
  if (x.rank() != 4 || y.rank() != 4) return GraphStatus::ErrorFatal;
  const bool params_native = scale.rank() == 1 && bias.rank() == 1 &&
                             mean.rank() == 1 && variance.rank() == 1;
  const bool params_backfilled =
      scale.rank() == 4 && bias.rank() == 4 && mean.rank() == 4 &&
      variance.rank() == 4 && scale.dim(0) == 1 && scale.dim(1) == 1 &&
      scale.dim(2) == 1 && bias.dim(0) == 1 && bias.dim(1) == 1 &&
      bias.dim(2) == 1 && mean.dim(0) == 1 && mean.dim(1) == 1 &&
      mean.dim(2) == 1 && variance.dim(0) == 1 && variance.dim(1) == 1 &&
      variance.dim(2) == 1;
  if (!params_native && !params_backfilled) return GraphStatus::ErrorFatal;

  const size_t channels = params_native ? scale.dim(0) : scale.dim(3);
  const bool nchw = x.dim(1) == channels && y.dim(1) == channels;
  const bool nhwc = x.dim(3) == channels && y.dim(3) == channels;
  if (channels == 0 || (!nchw && !nhwc)) return GraphStatus::ErrorFatal;
  const size_t batch = x.dim(0);
  const size_t height = nchw ? x.dim(2) : x.dim(1);
  const size_t width = nchw ? x.dim(3) : x.dim(2);
  const float epsilon_value = epsilon.rank() == 1
                                  ? float(epsilon(0))
                                  : float(epsilon(0, 0, 0, 0));
  if (!(epsilon_value > 0.0f)) return GraphStatus::ErrorFatal;
  y.set_dims(x);

  for (size_t batch_index = 0; batch_index < batch; ++batch_index) {
    for (size_t channel = 0; channel < channels; ++channel) {
      const float scale_value = params_native ? float(scale(channel))
                                              : float(scale(0, 0, 0, channel));
      const float bias_value = params_native ? float(bias(channel))
                                             : float(bias(0, 0, 0, channel));
      const float mean_value = params_native ? float(mean(channel))
                                             : float(mean(0, 0, 0, channel));
      const float variance_value = params_native
                                       ? float(variance(channel))
                                       : float(variance(0, 0, 0, channel));
      const float multiplier = scale_value / std::sqrt(variance_value + epsilon_value);
      for (size_t row = 0; row < height; ++row) {
        for (size_t column = 0; column < width; ++column) {
          const float value = nchw ? float(x(batch_index, channel, row, column))
                                   : float(x(batch_index, row, column, channel));
          const float result = (value - mean_value) * multiplier + bias_value;
          if (nchw) {
            y(batch_index, channel, row, column) = result;
          } else {
            y(batch_index, row, column, channel) = result;
          }
        }
      }
    }
  }
  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_DepthArtBatchNorm2d);
