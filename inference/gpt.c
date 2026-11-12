/* A GPT running inside a GPT.
 *
 * A character-level transformer -- attention, softmax, residual stream, autoregressive
 * decode -- whose forward pass is executed by the RV32IM machine compiled into the weight
 * matrices of Qwen Concept 27B. The inner model is the same architecture class as the
 * outer one, computed exactly, instruction by instruction.
 *
 * The machine is RV32IM_Zba: integer multiply and divide, no FPU. Measured on the machine
 * itself, a soft-float MAC costs 189 instructions against 4.22 for an int8 MAC, and expf
 * costs 2,185 -- 518 int8 MACs. A float softmax would therefore cost more than every
 * matmul in this model combined. So there is no floating point here at all:
 *
 *   - int8 weights, int32/int64 accumulators
 *   - requantization by a calibrated fixed-point multiplier: (acc * M0) >> shift.
 *     RV32M has MULH, so a 32x32 -> 64 bit product is two instructions.
 *   - RMSNorm with an exact integer floor-sqrt (no sqrtf). Normalization is
 *     scale-invariant, which is why RMSNorm quantizes cleanly here.
 *   - softmax by a 256-entry exponential lookup table (no expf)
 *   - one RV32M integer divide per attention output dimension
 *
 * Every operation mirrors int_forward() in train_gpt.py, so the Python specification and
 * this C agree by construction. g_expect holds the specification's generated tokens.
 */
#include <stdio.h>
#include <stdint.h>
#include "gpt_weights.h"
#include "gpt_expect.h"

/* How many tokens to generate. Each step replays the whole prefix, so the instruction
   count grows quadratically; override at build time to grade a shorter run on the
   checkpoint without retraining. */
#ifndef NGEN
#define NGEN GNGEN
#endif

/* Weights are const (DROM). Only these live in RAM. */
static int8_t kc[GL][GT][GD];
static int8_t vc[GL][GT][GD];
static int32_t x[GD];
static int8_t nbuf[GD];
static int8_t qv[GD], kv[GD], vv[GD], att[GD];
static int8_t hbuf[GFF];
static int32_t sc[GT];
static int32_t logits[GV];
static uint8_t seq[GT];

/* out = (acc * M0) >> shift, in 64-bit. */
static int64_t requant(int64_t acc, int idx)
{
    return (acc * (int64_t)g_m0[idx]) >> g_sh[idx];
}

static int8_t clamp8(int64_t v)
{
    if (v > 127) return 127;
    if (v < -127) return -127;
    return (int8_t)v;
}

/* Exact floor(sqrt(n)), bit by bit. Matches math.isqrt in the specification. */
static int32_t isqrt32(int64_t n)
{
    uint64_t v = (uint64_t)n, r = 0, b = 1ull << 62;
    if (n <= 0) return 0;
    while (b > v) b >>= 2;
    while (b) {
        if (v >= r + b) { v -= r + b; r = (r >> 1) + b; }
        else r >>= 1;
        b >>= 2;
    }
    return (int32_t)r;
}

/* RMSNorm, integer. int32 residual -> int8 normalized. */
static void rmsnorm_q(const int32_t *xv, const int8_t *g, int mul, int8_t *out)
{
    int64_t xs[GD], ss = 0;
    int32_t r;
    int64_t inv;
    for (int i = 0; i < GD; i++) {
        xs[i] = (int64_t)xv[i] >> RN_PRE;
        ss += xs[i] * xs[i];
    }
    r = isqrt32(ss / GD);
    if (r < 1) r = 1;
    inv = ((int64_t)1 << RN_K) / r;
    for (int i = 0; i < GD; i++)
        out[i] = clamp8(requant((xs[i] * inv) * (int64_t)g[i], mul));
}

/* int8 a[K] . int8 W[N][K] -> requantized int8[N]. */
static void mat_q(const int8_t *a, const int8_t *W, int K, int N, int mul, int8_t *out)
{
    for (int n = 0; n < N; n++) {
        const int8_t *w = &W[(int32_t)n * K];
        int32_t s = 0;
        for (int k = 0; k < K; k++) s += (int32_t)a[k] * (int32_t)w[k];
        out[n] = clamp8(requant(s, mul));
    }
}

/* Same, accumulating into the int32 residual stream (no int8 clamp). */
static void mat_add(const int8_t *a, const int8_t *W, int K, int N, int mul, int32_t *acc)
{
    for (int n = 0; n < N; n++) {
        const int8_t *w = &W[(int32_t)n * K];
        int32_t s = 0;
        for (int k = 0; k < K; k++) s += (int32_t)a[k] * (int32_t)w[k];
        acc[n] += (int32_t)requant(s, mul);
    }
}

/* One decode step at position pos for token tok; fills logits[]. */
static void forward(int tok, int pos)
{
    for (int i = 0; i < GD; i++) {
        int64_t e = (int64_t)g_tok[tok * GD + i]
                  + (((int64_t)g_pos[pos * GD + i] * POS_NUM) >> 10);
        x[i] = (int32_t)requant(e, MUL_EMB);
    }

    for (int l = 0; l < GL; l++) {
        const int8_t *Wq = &g_wq[(int32_t)l * GD * GD];
        const int8_t *Wk = &g_wk[(int32_t)l * GD * GD];
        const int8_t *Wv = &g_wv[(int32_t)l * GD * GD];
        const int8_t *Wo = &g_wo[(int32_t)l * GD * GD];
        const int8_t *W1 = &g_w1[(int32_t)l * GFF * GD];
        const int8_t *W2 = &g_w2[(int32_t)l * GD * GFF];
        int mn1 = (l == 0) ? MUL_N1_0 : MUL_N1_1;
        int mn2 = (l == 0) ? MUL_N2_0 : MUL_N2_1;
        int mq  = (l == 0) ? MUL_Q_0  : MUL_Q_1;
        int mk  = (l == 0) ? MUL_K_0  : MUL_K_1;
        int mv  = (l == 0) ? MUL_V_0  : MUL_V_1;
        int ms  = (l == 0) ? MUL_SC_0 : MUL_SC_1;
        int mo  = (l == 0) ? MUL_O_0  : MUL_O_1;
        int mh  = (l == 0) ? MUL_H_0  : MUL_H_1;
        int mm  = (l == 0) ? MUL_M_0  : MUL_M_1;

        rmsnorm_q(x, &g_gn1[l * GD], mn1, nbuf);
        mat_q(nbuf, Wq, GD, GD, mq, qv);
        mat_q(nbuf, Wk, GD, GD, mk, kv);
        mat_q(nbuf, Wv, GD, GD, mv, vv);
        for (int i = 0; i < GD; i++) { kc[l][pos][i] = kv[i]; vc[l][pos][i] = vv[i]; }

        for (int h = 0; h < GH; h++) {
            int off = h * GHD;
            int32_t mx;
            int64_t psum = 0;
            for (int t = 0; t <= pos; t++) {
                int64_t s = 0;
                for (int d = 0; d < GHD; d++)
                    s += (int64_t)qv[off + d] * (int64_t)kc[l][t][off + d];
                sc[t] = (int32_t)requant(s, ms);
            }
            mx = sc[0];
            for (int t = 1; t <= pos; t++) if (sc[t] > mx) mx = sc[t];
            for (int t = 0; t <= pos; t++) {
                int32_t d = mx - sc[t];
                if (d < 0) d = 0;
                if (d > 255) d = 255;
                sc[t] = g_exp[d];
                psum += sc[t];
            }
            if (psum < 1) psum = 1;
            for (int d = 0; d < GHD; d++) {
                int64_t acc = 0;
                for (int t = 0; t <= pos; t++)
                    acc += (int64_t)sc[t] * (int64_t)vc[l][t][off + d];
                att[off + d] = clamp8(acc / psum);   /* RV32M divide */
            }
        }
        mat_add(att, Wo, GD, GD, mo, x);

        rmsnorm_q(x, &g_gn2[l * GD], mn2, nbuf);
        for (int n = 0; n < GFF; n++) {
            const int8_t *w = &W1[(int32_t)n * GD];
            int32_t s = 0;
            int64_t q;
            for (int k = 0; k < GD; k++) s += (int32_t)nbuf[k] * (int32_t)w[k];
            q = requant(s, mh);
            if (q < 0) q = 0;                         /* ReLU */
            hbuf[n] = clamp8(q);
        }
        mat_add(hbuf, W2, GFF, GD, mm, x);
    }

    rmsnorm_q(x, g_gnf, MUL_NF, nbuf);
    for (int v = 0; v < GV; v++) {                    /* tied head */
        const int8_t *w = &g_tok[(int32_t)v * GD];
        int32_t s = 0;
        for (int k = 0; k < GD; k++) s += (int32_t)nbuf[k] * (int32_t)w[k];
        logits[v] = s;
    }
}

int main(void)
{
    int n = 0, ok = 1;

    for (int i = 0; i < GNPROMPT; i++) seq[n++] = g_prompt[i];

    for (int step = 0; step < NGEN; step++) {
        int best = 0;
        /* Replay the whole prefix each step, with a cleared cache: the KV entries are
           static, so a stale tail from the previous step would otherwise survive and
           the run would diverge from the specification. Exactness first; this program
           is a verification target, not a serving path. */
        for (int l = 0; l < GL; l++)
            for (int t = 0; t < GT; t++)
                for (int d = 0; d < GD; d++) { kc[l][t][d] = 0; vc[l][t][d] = 0; }
        for (int i = 0; i < n; i++) forward(seq[i], i);
        for (int v = 1; v < GV; v++) if (logits[v] > logits[best]) best = v;
        seq[n++] = (uint8_t)best;
    }

    printf("prompt=");
    for (int i = 0; i < GNPROMPT; i++) putchar(g_chars[seq[i]]);
    printf("\ngenerated=");
    for (int i = GNPROMPT; i < n; i++) putchar(g_chars[seq[i]]);
    putchar('\n');

    for (int i = 0; i < n; i++) if (seq[i] != g_expect[i]) ok = 0;
    printf("tokens=%d generated=%d match_spec=%d\n", n, NGEN, ok);
    printf("final_logit=%ld\n", (long)logits[seq[n - 1]]);
    return 0;
}
