#ifndef JPEG_FIRST_EOI_H
#define JPEG_FIRST_EOI_H
#include <stddef.h>
#include <stdint.h>

/* Incremental marker framing, not an entropy decoder. Caller retains an
 * immutable, contiguous prefix from byte zero and grows size monotonically.
 * Zero means incomplete or invalid (invalid is sticky); a positive result is
 * the exact first JPEG end, exclusive. No allocation and no rescanning. */
typedef struct {
    size_t pos, end, remaining;
    unsigned state, marker, length, saw_sof, saw_sos, invalid;
} jpeg_first_eoi_t;
enum { JE_SOI_FF, JE_SOI_D8, JE_MARK_FF, JE_MARK_CODE, JE_LEN_HI,
       JE_LEN_LO, JE_SKIP, JE_ENTROPY, JE_ENTROPY_FF };
static inline void jpeg_first_eoi_reset(jpeg_first_eoi_t *p) {
    jpeg_first_eoi_t z = {0}; *p = z;
}
static inline size_t jpeg_first_eoi_feed(jpeg_first_eoi_t *p,
                                        const uint8_t *buf, size_t size) {
    if (p->end) return p->end;
    if (p->invalid) return 0;
    if (size < p->pos || (!buf && size)) { p->invalid = 1; return 0; }
    while (p->pos < size && !p->invalid) {
        if (p->state == JE_SKIP) {
            size_t n = size - p->pos;
            if (n > p->remaining) n = p->remaining;
            p->pos += n; p->remaining -= n;
            if (!p->remaining)
                p->state = (p->marker == 0xDA || p->marker == 0xDC)
                    ? JE_ENTROPY : JE_MARK_FF;
            continue;
        }
        unsigned b = buf[p->pos++];
        switch (p->state) {
        case JE_SOI_FF:
            if (b != 0xFF) p->invalid = 1; else p->state = JE_SOI_D8;
            break;
        case JE_SOI_D8:
            if (b != 0xD8) p->invalid = 1; else p->state = JE_MARK_FF;
            break;
        case JE_MARK_FF:
            if (b != 0xFF) p->invalid = 1; else p->state = JE_MARK_CODE;
            break;
        case JE_ENTROPY:
            if (b == 0xFF) p->state = JE_ENTROPY_FF;
            break;
        case JE_ENTROPY_FF:
            if (b == 0xFF) break;
            if (b == 0 || (b >= 0xD0 && b <= 0xD7)) {
                p->state = JE_ENTROPY; break;
            }
            /* A real marker terminates the entropy segment. */
            /* fall through */
        case JE_MARK_CODE:
            if (b == 0xFF) break;
            if (b == 0xD9) {
                if (!p->saw_sof || !p->saw_sos) p->invalid = 1;
                else { p->end = p->pos; return p->end; }
            } else if (b == 0x01) {
                p->state = p->state == JE_ENTROPY_FF ? JE_ENTROPY : JE_MARK_FF;
            } else if (b < 0xC0 || b == 0xD8 || (b >= 0xD0 && b <= 0xD7)) {
                p->invalid = 1;
            } else {
                p->marker = b; p->state = JE_LEN_HI;
            }
            break;
        case JE_LEN_HI: p->length = b << 8; p->state = JE_LEN_LO; break;
        case JE_LEN_LO:
            p->length |= b;
            if (p->length < 2) { p->invalid = 1; break; }
            if (p->marker >= 0xC0 && p->marker <= 0xCF &&
                p->marker != 0xC4 && p->marker != 0xC8 && p->marker != 0xCC) {
                if (p->length < 11) { p->invalid = 1; break; }
                p->saw_sof = 1;
            }
            if (p->marker == 0xDA) {
                if (!p->saw_sof || p->length < 8) { p->invalid = 1; break; }
                p->saw_sos = 1;
            }
            p->remaining = p->length - 2;
            p->state = JE_SKIP;
            if (!p->remaining) p->state = JE_MARK_FF;
            break;
        default: p->invalid = 1; break;
        }
    }
    return 0;
}
#endif
