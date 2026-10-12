#pragma once
#include <stddef.h>
#include <stdint.h>

// Caller serializes publish/start/take with the sampler lock. A cursor is per
// consumer; the fixed ring never waits for consumers. Ordinals are independent
// of the raw uint32_t sensor sequence, including its rollover.
template<class Sample, size_t Capacity> class AtomImuHistory {
 public:
  static_assert(Capacity>0,"History must retain at least one sample");
  struct Cursor { uint64_t next=1, overwritten=0, expired=0; };
  void publish(const Sample& sample) {
    samples_[published_%Capacity]=sample;
    ++published_;
  }
  void startLatest(Cursor& cursor) const {
    cursor=Cursor();
    cursor.next=published_?published_:1;
  }
  bool take(Cursor& cursor,int64_t nowUs,int64_t maxAgeUs,Sample& sample) const {
    const uint64_t oldest=published_>=Capacity?published_-Capacity+1:1;
    if(cursor.next<oldest) {
      cursor.overwritten+=oldest-cursor.next;
      cursor.next=oldest;
    }
    // At most Capacity iterations even after an arbitrarily long stall.
    while(cursor.next<=published_) {
      const Sample& candidate=samples_[(cursor.next-1)%Capacity];
      ++cursor.next;
      if(nowUs-candidate.deviceUs>maxAgeUs) { ++cursor.expired;continue; }
      sample=candidate;
      return true;
    }
    return false;
  }
 private:
  Sample samples_[Capacity]={};
  uint64_t published_=0;
};
