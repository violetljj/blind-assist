#pragma once
#include <math.h>

// Gravity vector in the native, right-handed BMI270 sensor frame. This is a
// six-axis complementary filter, not an IMU-to-ToF extrinsic calibration.
// Specific force at rest points upward; body rotation changes it by -w cross g.
class AtomGravityFilter {
 public:
  static constexpr float tauSeconds = 0.5f;
  float gravity[3] = {0, 0, 1};
  bool initialized = false;
  bool accelCorrection = false;

  bool update(const float accelG[3], const float gyroDps[3], float dt,
              bool sampleGap) {
    float norm = sqrtf(accelG[0]*accelG[0] + accelG[1]*accelG[1] + accelG[2]*accelG[2]);
    accelCorrection = isfinite(norm) && norm >= 0.8f && norm <= 1.2f;
    if (!initialized || sampleGap) {
      // A gap is not integrated with a stale gyro sample. Re-anchor only when
      // specific force is compatible with gravity, otherwise mark pose invalid.
      initialized = accelCorrection;
      if (!initialized) return false;
      for (int i=0; i<3; ++i) gravity[i] = accelG[i]/norm;
      return true;
    }
    if (!(dt > 0) || !isfinite(dt)) return false;
    constexpr float radiansPerDegree = 0.017453292519943295f;
    float w[3];
    for (int i=0; i<3; ++i) {
      if (!isfinite(gyroDps[i])) return false;
      w[i] = gyroDps[i] * radiansPerDegree;
    }
    // Rodrigues rotation is exact for the constant angular rate over this dt,
    // avoiding Euler-angle singularities and cross-axis integration errors.
    float rate = sqrtf(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]);
    if (rate > 1e-7f) {
      float axis[3] = {w[0]/rate, w[1]/rate, w[2]/rate};
      float c=cosf(rate*dt), s=-sinf(rate*dt);
      float dot=axis[0]*gravity[0]+axis[1]*gravity[1]+axis[2]*gravity[2];
      float cross[3] = {
        axis[1]*gravity[2]-axis[2]*gravity[1],
        axis[2]*gravity[0]-axis[0]*gravity[2],
        axis[0]*gravity[1]-axis[1]*gravity[0]
      };
      for (int i=0; i<3; ++i) gravity[i]=gravity[i]*c+cross[i]*s+axis[i]*dot*(1-c);
    }
    if (accelCorrection) {
      float weight=dt/(tauSeconds+dt);
      for (int i=0; i<3; ++i) gravity[i]=(1-weight)*gravity[i]+weight*accelG[i]/norm;
    }
    float gNorm=sqrtf(gravity[0]*gravity[0]+gravity[1]*gravity[1]+gravity[2]*gravity[2]);
    if (!(gNorm>1e-6f) || !isfinite(gNorm)) { initialized=false; return false; }
    for (int i=0; i<3; ++i) gravity[i]/=gNorm;
    return true;
  }

  float pitchDeg() const {
    return atan2f(-gravity[0],sqrtf(gravity[1]*gravity[1]+gravity[2]*gravity[2]))*57.295779513f;
  }
  float rollDeg() const { return atan2f(gravity[1],gravity[2])*57.295779513f; }
};
