# Concept RV32IM 27B

Analysis code, merge tooling, inference demonstrations and evaluation artifacts for
*Exact Neural Network Inference Inside the Weights of Qwen Concept 27B*: an exact RV32IM
machine compiled into the weights of a 27B language model, and a neural network run on it.

Paper: https://concept.country/research/concept-rv32im

This is a port of [Qwenputer](https://github.com/divagr18/qwenputer) by Divyansh
Agrawal onto Concept 27B, the edited model from the
[companion work](https://github.com/conceptcountry/weight-space-programming).
The architecture, the instruction encoding and the original merge are his. See
NOTICE.

## What this is

A RISC-V RV32IM interpreter that exists as weights, not as code the model calls.
Eleven extra layers widen the hidden state from 5120 to 6656 and use the extra
dimensions as machine state. The model executes programs by running forward
passes.

Verified in the artifacts here:

- 2,648,144 instructions across 72 programs in standalone verification, and a sweep of
  61 programs and 5,094,103 instructions on the merged checkpoint with 61 of 61 exact.
- `calc` returning 121932631112635269 with zero wrong tokens.
- An MNIST classifier (784-32-10, int8) compiled to RV32IM and run inside the weights:
  2,059,619 instructions, machine stream exact against the reference emulator, all ten
  printed logits equal to the PyTorch model's.

A character-level GPT (110,976 int8 parameters) is compiled for the same machine and
reproduces its integer specification under QEMU. Its run on the checkpoint is not yet
complete and no checkpoint result is claimed for it.

## Reproducing the inference demonstrations

`inference/` holds everything needed. It requires a RISC-V cross toolchain with picolibc,
`qemu-riscv32`, PyTorch, and the Qwenputer `chipformer` package.

```
python inference/train_mlp.py OUT     # train, quantize to int8, emit mlp_weights.h
python inference/train_gpt.py OUT     # train, calibrate, emit gpt_weights.h
python inference/gen_expect.py OUT    # expected tokens derived from the shipped header
bash   inference/build.sh             # RV32IM ELFs
QWP_RESIDENT=11 python inference/c27_exec.py CHECKPOINT --elf c-mlp --out c-mlp.json
```

`bench_ops.c` measures what arithmetic costs on a machine with no FPU: an `expf` is 518
int8 multiply-accumulates, which is why both models are integer end to end.
`quant_probe.py` rounds the machine's layers onto lossy weight grids and re-runs programs;
every lossy grid tested breaks the machine, so released quantizations keep those eleven
layers in bf16 (`rv_quant.py`, `rv_dist_build.sh`).

## The merge, and two defects in it

`merge/` contains the tooling that found and fixed two real defects. The probes
are included because the measurements are the argument, not the conclusions
drawn from them.

**Greedy identity was 16/20.** Widening the hidden state with zeros shrinks the
RMS by sqrt(5120/6656), so every norm gain is rescaled by that factor. The factor
is irrational and the result was cast back to bfloat16, which rounds it:
**660,479 of 660,480 gains landed off the bfloat16 grid.** Exactly one survived.

Four fixes were measured before any was chosen:

| | approach | outcome |
|---|---|---|
| A | store rescaled norms in float32 | exact, +1.32 MB |
| B | pick a width where the scale is a power of two | exact, but 3.1x the checkpoint |
| C | exclude the chip dims from the variance | exact, but needs a custom RMSNorm |
| D | absorb the scale into the consumer projections | **14,116,454,400 of 14,116,454,400 values off-grid** |

D is the interesting failure: it shows the irrational factor is the problem, not
where you apply it.

A wins, and for a better reason than its cost. Stock Qwen3.5 `RMSNorm.forward`
calls `self.weight.float()` — it upcasts the gain to float32 on every forward
pass regardless of how it was stored. float32 is the dtype the module already
consumes; bfloat16 storage was discarding precision the runtime asks for one
line later.

The merge now asserts this rather than claiming it: if the final norm's host
rows are not bitwise equal to the exact arithmetic, the merge fails.

**Vision and MTP were dropped — 348 tensors.** Neither was broken. Both were
skipped by a single early-continue on the text-prefix test, with
`mtp_num_hidden_layers` pinned to zero. The vision tower runs in its own
1152-dimensional space and meets the text state only at `merger.linear_fc2`,
which the upstream `OUT_ROWS` pattern already matched — the rule existed and was
never reached.

## Layout

```
merge/patch_merge27.py      the three fixes, each with its measurement
merge/dryrun_merge.py       apply every shape/dtype rule in memory first
merge/probe_*.py            the measurements that chose the fix
emit_results.py             artifacts -> results/results.json
emit_numbers.py             results.json -> paper/numbers.typ
emit_inference.py           inference artifacts -> results/inference.json
emit_numbers_inference.py   inference.json -> paper/inference.typ
inference/                  the MNIST and GPT programs, the executor runner, the quantization probes
make_thumbnail.py           the noise-floor figure
artifacts/                  execution traces, teacher-forced logits, benchmarks
```

Set `CONCEPT_SRC` to your checkpoint directory before running the probes.

## On the failing criterion

Text identity against the unmerged host was 16/20 greedy-identical. Teacher
forced, the merged model agrees with the host on 955 of 960 positions — against
a harness self-agreement floor of 954/960. The disagreement sits at the noise
floor, which is why the paper argues the machine does not disturb the host
rather than claiming it provably cannot.

`merge/` is the attempt to remove that caveat at its source instead of arguing
around it.

## Weights

Published as [`ConceptCountry/concept-rv32im-27b`](https://huggingface.co/ConceptCountry/concept-rv32im-27b) on HuggingFace, Apache-2.0,
with the vision tower and the MTP head included. Mirrored on the Concept Country
registry: <https://models.concept.country/api/models/conceptcountry/concept-rv32im-27b>.

| folder | scheme | size | machine layers vs bf16 | text identity vs base |
|---|---|---|---|---|
| `bf16/` | bfloat16 | 89 GB | reference | 18/20 |
| `fp8/` | FP8_BLOCK | 58 GB | identical (156/156) | 14/20 |
| `nvfp4a16/` | NVFP4 W4A16 | 44 GB | identical (156/156) | 10/20 |
| `gguf/` | GGUF Q8_0 | 57 GB | bf16/f32 (157/157) | not measured |
| `gguf/` | GGUF Q4_K_M | 43 GB | bf16/f32 (157/157) | not measured |

In every quantized format the eleven machine layers (language-model layers 0-10), the
token embedding and the output head stay bf16; only the host's layers are quantized. A
weight-grid probe found that FP8, Q8_0 and Q4_K_M each take the machine from 4 of 4
programs exact to 0 of 4 when applied to its layers, while the bf16 control stays exact.

"machine layers vs bf16" held in 5 of 5 formats: for the safetensors
builds it is the sha256 of every tensor in that region against the bf16 release; for the
GGUF builds, which rename, fuse and transpose relative to the HuggingFace layout, it is
that no tensor in `blk.0`-`blk.10`, `token_embd` or `output` carries a quantized type.
Every format loads: the safetensors ones as the stock conditional-generation class, the
GGUF ones under `llama-cli`.

"text identity vs base" is greedy continuation of 20 prompts for 48 new
tokens against the *same format* of
[Qwen Concept 27B](https://huggingface.co/ConceptCountry/qwen-concept-27b), counting
identical continuations. At bf16 the release reproduces 18 of
20, with differences beginning a median of
121 characters in and reading as paraphrase. The two
GGUF rows were not measured: the runtime available here is a CPU build and the
comparison did not fit the machine's time budget. Those cells say so rather than
carrying an inferred number.

A sweep of the machine running programs on each quantized artifact has not been done.
What is established is that the machine's weights were not quantized.

Each folder carries a `SHA256SUMS`. The GGUF folder also carries
`mmproj-concept-rv32im-27b-bf16.gguf` for the vision tower. The GGUF builds exclude the
MTP head, as the base model's own GGUF does; it is present in the safetensors formats.

Experimental, under evaluation.

## License

Apache-2.0, as required by the upstream work. See LICENSE and NOTICE.
