#include "jpeg_first_eoi.h"
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
EXPORT size_t parse_chunks(const uint8_t *data, size_t size, size_t chunk) {
    jpeg_first_eoi_t p; jpeg_first_eoi_reset(&p);
    if (!chunk) return 0;
    for (size_t n = 0; n < size;) {
        size_t left = size - n; n += left < chunk ? left : chunk;
        size_t end = jpeg_first_eoi_feed(&p, data, n);
        if (end) return end;
    }
    return 0;
}
EXPORT size_t parse_split(const uint8_t *data, size_t size, size_t split) {
    jpeg_first_eoi_t p; jpeg_first_eoi_reset(&p);
    size_t first = jpeg_first_eoi_feed(&p, data, split);
    return first ? first : jpeg_first_eoi_feed(&p, data, size);
}
