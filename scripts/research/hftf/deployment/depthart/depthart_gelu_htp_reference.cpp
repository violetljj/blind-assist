// DepthART exact GELU HTP scalar float32 reference kernel.

#include <cmath>
#include <cstddef>

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

BEGIN_PKG_OP_DEFINITION(PKG_DepthArtGelu);

template <typename TensorType>
GraphStatus depthArtGeluReferenceImpl(TensorType &y, const TensorType &x);

DEF_PACKAGE_OP_AND_COST_AND_FLAGS(
    (depthArtGeluReferenceImpl<Tensor>), "DepthArtGelu", GLACIAL)

static inline float exactGelu(float value) {
  constexpr float kInverseSqrtTwo = 0.7071067811865475244f;
  return 0.5f * value * (1.0f + std::erf(value * kInverseSqrtTwo));
}

template <typename TensorType>
GraphStatus depthArtGeluReferenceImpl(TensorType &y, const TensorType &x) {
  if (x.rank() == 0 || x.rank() > 4 || y.rank() != x.rank()) {
    return GraphStatus::ErrorFatal;
  }
  y.set_dims(x);
  if (x.rank() == 1) {
    for (size_t i0 = 0; i0 < x.dim(0); ++i0) y(i0) = exactGelu(float(x(i0)));
  } else if (x.rank() == 2) {
    for (size_t i0 = 0; i0 < x.dim(0); ++i0)
      for (size_t i1 = 0; i1 < x.dim(1); ++i1)
        y(i0, i1) = exactGelu(float(x(i0, i1)));
  } else if (x.rank() == 3) {
    for (size_t i0 = 0; i0 < x.dim(0); ++i0)
      for (size_t i1 = 0; i1 < x.dim(1); ++i1)
        for (size_t i2 = 0; i2 < x.dim(2); ++i2)
          y(i0, i1, i2) = exactGelu(float(x(i0, i1, i2)));
  } else {
    for (size_t i0 = 0; i0 < x.dim(0); ++i0)
      for (size_t i1 = 0; i1 < x.dim(1); ++i1)
        for (size_t i2 = 0; i2 < x.dim(2); ++i2)
          for (size_t i3 = 0; i3 < x.dim(3); ++i3)
            y(i0, i1, i2, i3) = exactGelu(float(x(i0, i1, i2, i3)));
  }
  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_DepthArtGelu);
