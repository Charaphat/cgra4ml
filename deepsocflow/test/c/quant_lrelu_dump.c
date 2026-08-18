/* Dumps runtime.h's quant_lrelu function over a sweep, so the Python side can
 * be pinned against the C itself rather than against a re-derivation of it.
 *
 * quant_lrelu has never been exercised by any model built through this
 * backend so far (every deployed model has used relu/identity only), so
 * unlike div_round/quant_lut there is no prior transcription to check this
 * against - hand-deriving its shift semantics from reading the macro alone
 * is exactly the kind of thing this project's own history shows is easy to
 * get subtly wrong (e.g. the asymmetric clip bound below is not obvious from
 * a first read). Transcribed verbatim from deepsocflow/c/runtime.h and
 * compared, not reimplemented.
 *
 *   cc -o quant_lrelu_dump quant_lrelu_dump.c && ./quant_lrelu_dump
 *
 * Prints "x nzero shift pl_scale x_bits result" per line. Consumed by
 * deepsocflow/test/py/test_brevitas_conv.py::test_quant_lrelu_matches_c.
 */
#include <stdio.h>
#include <stdint.h>

typedef int8_t  i8;
typedef int32_t i32;

/* verbatim from deepsocflow/c/runtime.h */
#define shift_round(n, s) (((n) + ((s)>0 ? (1<<((s)-1)) - (~((n)>>(s))&1) : 0)) >> s)
#define clip(x, xmin, xmax) (((x) < (xmin)) ? (xmin) : ((x) > (xmax)) ? (xmax) : (x))

static inline i32 quant_lrelu(i32 x, i8 nzero, i8 shift, i8 pl_scale, i8 X_BITS){
  x = x < 0 ? (nzero ? x: 0) : x << pl_scale;
  x = shift_round(x, shift);
  x = clip(x, -(1<<(X_BITS-pl_scale-1)), (1<<(X_BITS-1))-1);
  return x;
}

int main(void) {
    const i8 x_bits_all[] = {8};
    /* pl_scale: 0 (relu/identity) through 4 (slope 2**-4). shift always
     * includes pl_scale (shift = pl_scale + acc_frac - act_frac in the real
     * pipeline), swept across a representative range of acc-vs-act frac
     * deltas. nzero=0 is relu, nzero=1 is identity/leaky_relu. */
    for (int xbi = 0; xbi < 1; xbi++) {
        i8 X_BITS = x_bits_all[xbi];
        for (i8 pl_scale = 0; pl_scale <= 4; pl_scale++) {
            for (i8 extra_shift = 0; extra_shift <= 6; extra_shift++) {
                i8 shift = pl_scale + extra_shift;
                for (i8 nzero = 0; nzero <= 1; nzero++) {
                    /* Sweep x over a range wide enough to cover the
                     * accumulator's real magnitude at these fracs - the
                     * pre-shift-round value, not the post-clip one. */
                    for (i32 x = -1 << 16; x <= (1 << 16); x += 37) {
                        i32 result = quant_lrelu(x, nzero, shift, pl_scale, X_BITS);
                        printf("%d %d %d %d %d %d\n", x, nzero, shift, pl_scale, X_BITS, result);
                    }
                }
            }
        }
    }
    return 0;
}
