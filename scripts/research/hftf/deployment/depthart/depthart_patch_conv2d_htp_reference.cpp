// DepthART first patch convolution HTP scalar float32 reference kernel.
//
// This is intentionally correctness-first and supports only the frozen
// 1x3x448x448 -> 1x24x224x224, kernel=3, stride=2, pad=1 contract.

#include <cstddef>

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

BEGIN_PKG_OP_DEFINITION(PKG_DepthArtPatchConv2d);

template <typename TensorType>
GraphStatus depthArtPatchConv2dReferenceImpl(TensorType &y,
                                             const TensorType &x,
                                             const TensorType &weight);

DEF_PACKAGE_OP_AND_COST_AND_FLAGS(
    (depthArtPatchConv2dReferenceImpl<Tensor>), "DepthArtPatchConv2d", GLACIAL)

template <typename TensorType>
GraphStatus depthArtPatchConv2dReferenceImpl(TensorType &y,
                                             const TensorType &x,
                                             const TensorType &weight) {
  if (x.rank() != 4 || weight.rank() != 4 || y.rank() != 4) {
    return GraphStatus::ErrorFatal;
  }

  const bool x_nchw = x.dim(0) == 1 && x.dim(1) == 3 &&
                      x.dim(2) == 448 && x.dim(3) == 448;
  const bool x_nhwc = x.dim(0) == 1 && x.dim(1) == 448 &&
                      x.dim(2) == 448 && x.dim(3) == 3;
  const bool weight_oihw = weight.dim(0) == 24 && weight.dim(1) == 3 &&
                           weight.dim(2) == 3 && weight.dim(3) == 3;
  const bool weight_hwio = weight.dim(0) == 3 && weight.dim(1) == 3 &&
                           weight.dim(2) == 3 && weight.dim(3) == 24;
  const bool y_nchw = y.dim(0) == 1 && y.dim(1) == 24 &&
                      y.dim(2) == 224 && y.dim(3) == 224;
  const bool y_nhwc = y.dim(0) == 1 && y.dim(1) == 224 &&
                      y.dim(2) == 224 && y.dim(3) == 24;
  if ((!x_nchw && !x_nhwc) || (!weight_oihw && !weight_hwio) ||
      (!y_nchw && !y_nhwc)) {
    return GraphStatus::ErrorFatal;
  }

  for (size_t output_y = 0; output_y < 224; ++output_y) {
    for (size_t output_x = 0; output_x < 224; ++output_x) {
      for (size_t output_channel = 0; output_channel < 24; ++output_channel) {
        float accumulator = 0.0f;
        for (size_t input_channel = 0; input_channel < 3; ++input_channel) {
          for (size_t kernel_y = 0; kernel_y < 3; ++kernel_y) {
            const int input_y = int(output_y * 2 + kernel_y) - 1;
            if (input_y < 0 || input_y >= 448) continue;
            for (size_t kernel_x = 0; kernel_x < 3; ++kernel_x) {
              const int input_x = int(output_x * 2 + kernel_x) - 1;
              if (input_x < 0 || input_x >= 448) continue;
              const float input_value = x_nchw
                  ? float(x(0, input_channel, size_t(input_y), size_t(input_x)))
                  : float(x(0, size_t(input_y), size_t(input_x), input_channel));
              const float weight_value = weight_oihw
                  ? float(weight(output_channel, input_channel, kernel_y, kernel_x))
                  : float(weight(kernel_y, kernel_x, input_channel, output_channel));
              accumulator += input_value * weight_value;
            }
          }
        }
        if (y_nchw) {
          y(0, output_channel, output_y, output_x) = accumulator;
        } else {
          y(0, output_y, output_x, output_channel) = accumulator;
        }
      }
    }
  }
  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_DepthArtPatchConv2d);
