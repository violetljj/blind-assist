#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <initializer_list>
#include "atom_imu_filter.h"
#include "atom_bmi270_config.h"
static int cases=0;
static void check(bool ok,const char* name){++cases;if(!ok){std::fprintf(stderr,"FAIL %s\n",name);std::exit(1);}}
int main(){
  const float zero[3]={0,0,0},rest[3]={0,0,1};
  AtomGravityFilter f;
  check(f.update(rest,zero,0,false),"initial rest valid");
  check(std::abs(f.pitchDeg())<1e-4 && std::abs(f.rollDeg())<1e-4,"rest zero");
  for(float degrees: {-30.f,30.f}){
    float r=degrees*0.017453292519943295f;
    const float ap[3]={-std::sin(r),0,std::cos(r)};
    AtomGravityFilter p;p.update(ap,zero,0,false);
    check(std::abs(p.pitchDeg()-degrees)<1e-3 && std::abs(p.rollDeg())<1e-3,"pitch accel +/-30");
    const float ar[3]={0,std::sin(r),std::cos(r)};
    AtomGravityFilter roll;roll.update(ar,zero,0,false);
    check(std::abs(roll.rollDeg()-degrees)<1e-3 && std::abs(roll.pitchDeg())<1e-3,"roll accel +/-30");
    const float noCorrection[3]={0,0,2},gy[3]={0,degrees,0},gx[3]={degrees,0,0};
    AtomGravityFilter ip;ip.update(rest,zero,0,false);
    for(int n=0;n<100;++n)ip.update(noCorrection,gy,0.01,false);
    check(std::abs(ip.pitchDeg()-degrees)<0.002 && !ip.accelCorrection,"gyro pitch sign +/-30");
    AtomGravityFilter ir;ir.update(rest,zero,0,false);
    for(int n=0;n<100;++n)ir.update(noCorrection,gx,0.01,false);
    check(std::abs(ir.rollDeg()-degrees)<0.002,"gyro roll sign +/-30");
  }
  const float biased[3]={0,1,0};
  for(int n=0;n<6000;++n)f.update(rest,biased,0.01,false);
  check(std::abs(f.pitchDeg())<0.51 && std::abs(f.rollDeg())<0.001,"stationary1dps bias bounded by accel feedback");
  const float force[3]={0,0,2};
  check(!f.update(force,zero,0.1,true),"gap under acceleration invalid");
  check(f.update(rest,zero,0.01,false) && std::abs(f.pitchDeg())<1e-4,"re-anchor after invalid gap");
  check(!f.update(rest,zero,0,false),"zero dt rejected");
  std::printf("PASS %d filter checks; configuration_bytes=%zu\n",cases,sizeof(atomBmi270Config));
}
