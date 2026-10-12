#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "atom_imu_history.h"
struct Sample { uint32_t seq=0;int64_t deviceUs=0;int raw=0; };
static int cases=0;
static void check(bool ok,const char* name) {
  ++cases;if(!ok){std::fprintf(stderr,"FAIL %s\n",name);std::exit(1);}
}
int main() {
  using History=AtomImuHistory<Sample,64>;
  History h;History::Cursor c;Sample s;
  h.startLatest(c);
  check(!h.take(c,0,500000,s),"empty queue");
  for(int i=1;i<=20;++i)h.publish({(uint32_t)i,i*10000,i*7});
  h.startLatest(c);
  check(h.take(c,200000,500000,s)&&s.seq==20,"connection starts latest, no old backlog");
  // Sender holds a copied frame while the sampler continues. Serializing and
  // writing fragments must not substitute a newer sample midway through JSON.
  std::string pending="{\"seq\":"+std::to_string(s.seq)+",\"device_us\":"+std::to_string(s.deviceUs)+",\"raw\":"+std::to_string(s.raw)+"}\n";
  std::string wire=pending.substr(0,7);
  for(int i=21;i<=40;++i)h.publish({(uint32_t)i,i*10000,i*7});
  wire+=pending.substr(7);
  check(wire=="{\"seq\":20,\"device_us\":200000,\"raw\":140}\n","partial packet remains exact during200ms sampling");
  for(int i=21;i<=40;++i) {
    check(h.take(c,400000,500000,s)&&s.seq==(uint32_t)i&&s.deviceUs==i*10000&&s.raw==i*7,"200ms stall drains every original sample in order");
  }
  check(!h.take(c,400000,500000,s)&&c.overwritten==0&&c.expired==0,"no duplicates or losses after catchup");
  History::Cursor stalled=c;
  for(int i=41;i<=140;++i)h.publish({(uint32_t)i,i*10000,i*7});
  check(h.take(c,1400000,500000,s)&&s.seq==90,"long stall skips overwritten and expired records");
  check(c.overwritten==36&&c.expired==13,"separate exact overflow and age loss counts");
  check(h.take(stalled,1400000,500000,s)&&s.seq==90&&stalled.overwritten==36,"independent consumers");
  h.startLatest(c);
  check(h.take(c,1400000,500000,s)&&s.seq==140&&c.overwritten==0&&c.expired==0,"reconnection starts current, counters reset per consumer");
  h.publish({141,1410000,987});
  check(!h.take(c,2000000,500000,s)&&c.expired==1,"stopped sensor stale sample rejected");
  h.publish({142,2010000,321});
  check(h.take(c,2010000,500000,s)&&s.seq==142&&s.raw==321,"fresh sample after stale period");
  History wrap;History::Cursor wc;
  wrap.publish({UINT32_MAX,10000,1});wrap.startLatest(wc);
  check(wrap.take(wc,10000,500000,s)&&s.seq==UINT32_MAX,"sequence before rollover");
  wrap.publish({0,20000,2});wrap.publish({1,30000,3});
  check(wrap.take(wc,30000,500000,s)&&s.seq==0&&s.deviceUs==20000,"raw sequence rollover retained");
  check(wrap.take(wc,30000,500000,s)&&s.seq==1,"sequence continues after rollover");
  // A pending copy survives history overwrite; the next take accounts only
  // for unselected records. This is the transport's partial-send ownership.
  History::Cursor pc;wrap.startLatest(pc);wrap.take(pc,30000,500000,s);
  const Sample pendingSample=s;
  for(int i=2;i<=100;++i)wrap.publish({(uint32_t)i,(i+2)*10000,i});
  check(pendingSample.seq==1&&pendingSample.deviceUs==30000&&pendingSample.raw==3,"pending frame survives ring overwrite");
  check(wrap.take(pc,1020000,500000,s)&&s.seq==50&&pc.overwritten==35&&pc.expired==13,"loss accounting excludes immutable pending frame");
  History independent;History::Cursor fast,slow;
  independent.startLatest(fast);independent.startLatest(slow);
  for(int i=1;i<=100;++i) {
    independent.publish({(uint32_t)i,i*10000,i});
    check(independent.take(fast,i*10000,500000,s)&&s.seq==(uint32_t)i,"slow client does not drop fast client's samples");
  }
  check(fast.overwritten==0&&fast.expired==0,"fast client no losses");
  check(independent.take(slow,1000000,500000,s)&&s.seq==50&&slow.overwritten==36&&slow.expired==13,"slow client bounded backlog independent of fast");
  History boundary;History::Cursor bc;
  boundary.publish({1,10000,1});boundary.startLatest(bc);
  check(boundary.take(bc,510000,500000,s)&&s.seq==1,"age cap is inclusive");
  boundary.publish({2,20000,2});
  check(!boundary.take(bc,520001,500000,s)&&bc.expired==1,"one microsecond over age cap excluded");
  std::printf("PASS %d IMU history checks\n",cases);
}
