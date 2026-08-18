/* Dumps runtime.h's write_x X_BITS>8 branch (the direct i16 store, see
 * write_x's #else path in deepsocflow/c/runtime.h) over a sweep of values, so
 * the Python side (dataflow.py::pack_words_into_bytes's bits>8 branch) can be
 * pinned against the real compiled C rather than an assumption that the two
 * use the same byte order. Both sides are "just" a native int16 store -
 * but that native order is exactly the kind of thing that's silently
 * platform-dependent and easy to get backwards when one side is numpy's
 * .tobytes() and the other is a raw C pointer cast, so it's worth pinning
 * for real rather than trusting they agree.
 *
 *   cc -o write_x_wide_dump write_x_wide_dump.c && ./write_x_wide_dump
 *
 * Prints one line per input value: "value byte0 byte1" (bytes as unsigned
 * 0-255). Consumed by
 * deepsocflow/test/py/test_brevitas_conv.py::test_write_x_wide_matches_c.
 */
#include <stdio.h>
#include <stdint.h>

typedef int16_t i16;
typedef uint8_t u8;

/* verbatim from deepsocflow/c/runtime.h's write_x, X_BITS>8 branch:
 *   i32 byte_idx = flat_index * X_BYTES_PER_WORD;
 *   write_flush_i16((i16*)(p_out_buffer + byte_idx), (i16)val);
 * - collapsed here to a single word (flat_index=0), since the sweep is over
 * VALUES, not addressing (the addressing math is already X_BITS-generic and
 * shared with the X_BITS<=8 path's own flat_index computation, which has its
 * own coverage elsewhere). */
static inline void write_flush_i16(i16 *addr, i16 val) {
  *addr = val;
}

int main(void) {
    /* Full signed 16-bit range, sampled (not exhaustive - 65536 lines is
     * plenty without printing all of it), plus the exact boundary values. */
    i16 values[] = {
        0, 1, -1, 2, -2, 100, -100, 1000, -1000,
        32767, 32766, -32768, -32767, 12345, -12345, 256, -256, 257, -257
    };
    int n = (int)(sizeof(values) / sizeof(values[0]));

    for (int i = 0; i < n; i++) {
        u8 buf[2] = {0, 0};
        write_flush_i16((i16*)buf, values[i]);
        printf("%d %u %u\n", (int)values[i], (unsigned)buf[0], (unsigned)buf[1]);
    }
    return 0;
}
