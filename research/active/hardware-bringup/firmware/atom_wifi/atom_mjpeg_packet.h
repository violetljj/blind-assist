#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

// One owned multipart part, reused only after its synchronous send completes.
// JPEG bytes are copied unchanged; the camera framebuffer can then be returned.
class AtomMjpegPacket {
 public:
  static constexpr size_t headerCapacity = 768;
  static constexpr size_t maxJpegBytes = 2 * 1024 * 1024;
  using Resize = void* (*)(void*, size_t);
  using Release = void (*)(void*);
  AtomMjpegPacket(Resize resize, Release release):resize_(resize),release_(release) {}
  ~AtomMjpegPacket() { if(buffer_)release_(buffer_); }
  AtomMjpegPacket(const AtomMjpegPacket&) = delete;
  AtomMjpegPacket& operator=(const AtomMjpegPacket&) = delete;

  bool copyJpeg(const uint8_t* jpeg,size_t length) {
    size_=0;jpegLength_=0;
    if(!jpeg || !length || length>maxJpegBytes)return false;
    const size_t required=headerCapacity+length;
    if(required>capacity_) {
      size_t next=capacity_?capacity_:32768;
      while(next<required) {
        const size_t limit=headerCapacity+maxJpegBytes;
        next=next>limit/2?limit:next*2;
      }
      void* replacement=resize_(buffer_,next);
      if(!replacement)return false; // Existing allocation is still owned.
      buffer_=static_cast<uint8_t*>(replacement);capacity_=next;
    }
    std::memcpy(buffer_+headerCapacity,jpeg,length);
    jpegLength_=length;
    return true;
  }
  bool prependHeader(const char* header,size_t length) {
    size_=0;
    if(!buffer_ || !jpegLength_ || !header || !length || length>=headerCapacity)return false;
    offset_=headerCapacity-length;
    std::memcpy(buffer_+offset_,header,length);
    size_=length+jpegLength_;
    return true;
  }
  const char* data() const { return buffer_?reinterpret_cast<const char*>(buffer_+offset_):nullptr; }
  size_t size() const { return size_; }

 private:
  Resize resize_;Release release_;
  uint8_t* buffer_=nullptr;
  size_t capacity_=0,jpegLength_=0,offset_=0,size_=0;
};
