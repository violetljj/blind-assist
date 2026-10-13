// Task-owned Arduino Wire transport for the supplied ST VL53LMZ ULD.
#include <Arduino.h>
#include <Wire.h>
#include "esp32-hal-i2c.h"
#include "platform.h"
static int8_t nativeReadBus = -1;
static uint32_t nativeReadClockHz = 0;
static uint32_t nativeReadTransferUs = 0;
extern "C" {
uint8_t UseEsp32HalRead(uint8_t bus) {
  if (bus>=SOC_I2C_NUM || !i2cIsInit(bus)) return 1;
  uint32_t clockHz = 0;
  if (i2cGetClock(bus,&clockHz)!=ESP_OK || !clockHz) return 1;
  nativeReadClockHz = clockHz;
  nativeReadBus = bus;
  return 0;
}
uint32_t ReadTransferUs() { return nativeReadTransferUs; }
uint8_t WrMulti(VL53LMZ_Platform* p, uint16_t reg, uint8_t* bytes, uint32_t n) {
  while (n) {
    size_t chunk = n > 120 ? 120 : n;
    Wire.beginTransmission(p->address >> 1);
    Wire.write(uint8_t(reg >> 8)); Wire.write(uint8_t(reg));
    if (Wire.write(bytes, chunk) != chunk) { Wire.endTransmission(); return 1; }
    if (Wire.endTransmission() != 0) return 1;
    reg += chunk; bytes += chunk; n -= chunk;
  }
  return 0;
}
uint8_t RdMulti(VL53LMZ_Platform* p, uint16_t reg, uint8_t* bytes, uint32_t n) {
  if (nativeReadBus>=0 && n) {
    const uint8_t addressBytes[] = {uint8_t(reg >> 8), uint8_t(reg)};
    // Include both slave addresses, two register bytes and each ACK/NACK bit.
    const uint64_t wireMs = (((uint64_t)n+4)*9*1000+nativeReadClockHz-1)/nativeReadClockHz;
    uint64_t timeoutMs = 2*wireMs+25;
    if (timeoutMs<100) timeoutMs = 100;
    // ESP-IDF's transfer timeout is a signed int in milliseconds.
    if (timeoutMs>INT32_MAX) return 1;
    size_t readCount = 0;
    const uint32_t started = micros();
    const esp_err_t status = i2cWriteReadNonStop(uint8_t(nativeReadBus), p->address >> 1,
        addressBytes, sizeof(addressBytes), bytes, n, (uint32_t)timeoutMs, &readCount);
    nativeReadTransferUs += (uint32_t)(micros()-started);
    return status==ESP_OK && readCount==n ? 0 : 1;
  }
  while (n) {
    size_t chunk = n > 120 ? 120 : n;
    Wire.beginTransmission(p->address >> 1);
    Wire.write(uint8_t(reg >> 8)); Wire.write(uint8_t(reg));
    if (Wire.endTransmission(false) != 0) return 1;
    if (Wire.requestFrom(uint8_t(p->address >> 1), chunk) != chunk) return 1;
    for (size_t i=0; i<chunk; ++i) bytes[i] = Wire.read();
    reg += chunk; bytes += chunk; n -= chunk;
  }
  return 0;
}
uint8_t RdByte(VL53LMZ_Platform* p, uint16_t r, uint8_t* b) { return RdMulti(p,r,b,1); }
uint8_t WrByte(VL53LMZ_Platform* p, uint16_t r, uint8_t b) { return WrMulti(p,r,&b,1); }
void SwapBuffer(uint8_t* b, uint16_t n) {
  for (uint16_t i=0; i+3<n; i+=4) {
    uint8_t t=b[i]; b[i]=b[i+3]; b[i+3]=t;
    t=b[i+1]; b[i+1]=b[i+2]; b[i+2]=t;
  }
}
uint8_t WaitMs(VL53LMZ_Platform*, uint32_t ms) { delay(ms); return 0; }
}
