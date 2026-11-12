/* A neural network running inside a neural network.
 *
 * An MNIST classifier (784 -> 32 -> 10, int8 weights, int32 accumulators) whose forward
 * pass is executed by the RV32IM machine compiled into the weight matrices of Qwen Concept
 * 27B. Every instruction of this inference is a state transition of a transformer's
 * forward pass, verified token-for-token against a reference emulator.
 *
 * Integer only: the machine is RV32IM_Zba and has no FPU, so there is no float anywhere
 * here, not even hidden in libgcc. Requantization between layers is an arithmetic right
 * shift, chosen at training time to be a power of two.
 *
 * Memory: weights and test images are const, so they are linked into the 128 KB DROM.
 * RAM holds only the 32 int32 layer-1 accumulators, 32 int8 activations and 10 int32
 * logits -- well under 1 KB of the 32 KB available.
 */
#include <stdio.h>
#include <stdint.h>
#include "mlp_weights.h"

static int32_t acc1[MLP_HID];
static int8_t hid[MLP_HID];
static int32_t logits[MLP_OUT];

/* One forward pass over a 784-pixel uint8 image. Returns the argmax class. */
static int mlp_forward(const uint8_t *img)
{
    /* Layer 1: int8 x int8 -> int32. Input pixels are halved to 0..127 so the
       accumulator cannot leave int32: 784 * 127 * 127 = 12,644,416. */
    for (int j = 0; j < MLP_HID; j++) {
        const int8_t *w = &mlp_w1[j * MLP_IN];
        int32_t s = mlp_b1[j];
        for (int i = 0; i < MLP_IN; i++)
            s += (int32_t)(img[i] >> 1) * (int32_t)w[i];
        acc1[j] = s;
    }

    /* ReLU and requantize to int8 by a power-of-two shift. */
    for (int j = 0; j < MLP_HID; j++) {
        int32_t v = acc1[j] >> MLP_SHIFT1;
        if (v < 0) v = 0;
        if (v > 127) v = 127;
        hid[j] = (int8_t)v;
    }

    /* Layer 2. */
    int best = 0;
    for (int k = 0; k < MLP_OUT; k++) {
        const int8_t *w = &mlp_w2[k * MLP_HID];
        int32_t s = mlp_b2[k];
        for (int j = 0; j < MLP_HID; j++)
            s += (int32_t)hid[j] * (int32_t)w[j];
        logits[k] = s;
        if (s > logits[best]) best = k;
    }
    return best;
}

int main(void)
{
    int correct = 0;

    for (int t = 0; t < MLP_NTEST; t++) {
        const uint8_t *img = &mlp_test_img[t * MLP_IN];
        int pred = mlp_forward(img);
        int want = mlp_test_lab[t];
        if (pred == want) correct++;
        /* Print the argmax and the winning logit so the output is sensitive to every
           accumulator bit, not just to the classification being right. */
        printf("img%d want=%d pred=%d logit=%ld\n", t, want, pred, (long)logits[pred]);
    }

    printf("correct=%d/%d\n", correct, MLP_NTEST);
    return 0;
}
