#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "atom_mjpeg_packet.h"
static int checks=0,releases=0,allocations=0;
static bool rejectAllocation=false;
static size_t allocatedBytes=0;
static void check(bool condition,const char* name) {
  ++checks;if(!condition){std::fprintf(stderr,"FAIL: %s\n",name);std::exit(1);}
}
static void* resize(void* pointer,size_t bytes) {
  ++allocations;allocatedBytes=bytes;
  return rejectAllocation?nullptr:std::realloc(pointer,bytes);
}
static void release(void* pointer){++releases;std::free(pointer);}
int main() {
  {
    AtomMjpegPacket packet(resize,release);
    std::vector<uint8_t> jpeg={0xff,0xd8,0,1,0xff,0xd9};
    const std::string header="\r\n--BAFRAME\r\nContent-Length: 6\r\n\r\n";
    check(packet.copyJpeg(jpeg.data(),jpeg.size()),"copy exact binary JPEG");
    std::fill(jpeg.begin(),jpeg.end(),0); // Simulate returning/reusing the camera framebuffer.
    check(packet.prependHeader(header.data(),header.size()),"prepend multipart header");
    const std::string expected=header+std::string("\xff\xd8\0\1\xff\xd9",6);
    check(std::string(packet.data(),packet.size())==expected,"owned packet survives camera overwrite");
    check(allocatedBytes==32768 && allocations==1,"one bounded reusable initial allocation");
    check(packet.copyJpeg(reinterpret_cast<const uint8_t*>("xyz"),3) &&
      packet.prependHeader(header.data(),header.size()) && allocations==1,"reuse allocation without frame queue");
    std::vector<uint8_t> large(40000,0xa5);
    rejectAllocation=true;
    check(!packet.copyJpeg(large.data(),large.size()) && packet.size()==0,"allocation failure cannot resend old frame");
    check(!packet.prependHeader(header.data(),header.size()),"failed JPEG cannot be finalized");
    rejectAllocation=false;
    check(packet.copyJpeg(large.data(),large.size()) && allocatedBytes==65536,"growth after failed realloc");
    check(!packet.prependHeader(header.data(),AtomMjpegPacket::headerCapacity) && packet.size()==0,"reject overflowing header");
    check(packet.prependHeader(header.data(),header.size()) && packet.size()==40000+header.size(),"exact grown packet length");
    check(!packet.copyJpeg(large.data(),AtomMjpegPacket::maxJpegBytes+1) && packet.size()==0,"reject oversized frame before copy");
  }
  check(releases==1,"owned allocation released once on stream exit");
  {
    rejectAllocation=true;AtomMjpegPacket packet(resize,release);
    const uint8_t jpeg=1;
    check(!packet.copyJpeg(&jpeg,1) && packet.data()==nullptr,"first allocation failure remains empty");
  }
  check(releases==1,"no release for failed initial allocation");
  std::printf("PASS: %d packet integrity and ownership checks\n",checks);
}
