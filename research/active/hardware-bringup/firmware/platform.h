#ifndef XIAO_TOF_PLATFORM_H
#define XIAO_TOF_PLATFORM_H
#include <stdint.h>
#include <string.h>
#define VL53LMZ_NB_TARGET_PER_ZONE 1U
// Keep the ULD's conversions enabled: distance_mm is millimetres.
typedef struct { uint16_t address; } VL53LMZ_Platform;
#ifdef __cplusplus
extern "C" {
#endif
// Opt in only after Wire.begin succeeds on this dedicated bus. The default
// remains Wire with 120-byte chunks; no unfinished Wire transaction may exist.
// Native timeout is max(100 ms, 2 * ceil(wire time in ms) + 25 ms).
uint8_t UseEsp32HalRead(uint8_t);
// Cumulative native transfer elapsed time, including HAL/bus waits; wraps at 2^32.
uint32_t ReadTransferUs(void);
uint8_t RdByte(VL53LMZ_Platform*, uint16_t, uint8_t*);
uint8_t WrByte(VL53LMZ_Platform*, uint16_t, uint8_t);
uint8_t RdMulti(VL53LMZ_Platform*, uint16_t, uint8_t*, uint32_t);
uint8_t WrMulti(VL53LMZ_Platform*, uint16_t, uint8_t*, uint32_t);
void SwapBuffer(uint8_t*, uint16_t);
uint8_t WaitMs(VL53LMZ_Platform*, uint32_t);
#ifdef __cplusplus
}
#endif
#endif
