/* Micro-benchmark: what does arithmetic actually cost on the in-weights RV32IM machine?
 *
 * Built once per MODE; the reference emulator reports exact instruction counts, so the
 * difference between MODE 0 and the others gives a real per-operation cost instead of a
 * guess. This decides whether a transformer in the weights must be integer-only or
 * whether soft-float is affordable.
 *
 *   MODE 0  baseline loop          MODE 2  float32 multiply-accumulate (soft-float)
 *   MODE 1  int8 multiply-accumulate   MODE 3  expf
 *   MODE 4  float32 divide           MODE 5  sqrtf
 */
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#ifndef N
#define N 10000
#endif
#ifndef MODE
#define MODE 0
#endif

static int8_t ia[256], ib[256];
static float fa[256], fb[256];

int main(void)
{
    for (int i = 0; i < 256; i++) {
        ia[i] = (int8_t)(i * 7);
        ib[i] = (int8_t)(i * 13);
        fa[i] = (float)i * 0.01f + 0.5f;
        fb[i] = (float)i * 0.02f + 0.25f;
    }

#if MODE == 0
    volatile int32_t acc = 0;
    for (int i = 0; i < N; i++) acc += (int32_t)(i & 255);
    printf("base %ld\n", (long)acc);

#elif MODE == 1
    volatile int32_t acc = 0;
    for (int i = 0; i < N; i++) acc += (int32_t)ia[i & 255] * (int32_t)ib[i & 255];
    printf("imac %ld\n", (long)acc);

#elif MODE == 2
    volatile float acc = 0.0f;
    for (int i = 0; i < N; i++) acc += fa[i & 255] * fb[i & 255];
    printf("fmac %ld\n", (long)acc);

#elif MODE == 3
    volatile float acc = 0.0f;
    for (int i = 0; i < N; i++) acc += expf(fa[i & 255]);
    printf("fexp %ld\n", (long)acc);

#elif MODE == 4
    volatile float acc = 0.0f;
    for (int i = 0; i < N; i++) acc += fa[i & 255] / fb[i & 255];
    printf("fdiv %ld\n", (long)acc);

#elif MODE == 5
    volatile float acc = 0.0f;
    for (int i = 0; i < N; i++) acc += sqrtf(fa[i & 255]);
    printf("fsqrt %ld\n", (long)acc);
#endif

    return 0;
}
