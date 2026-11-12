#import "numbers.typ": *
#import "inference.typ": *

#set document(title: "Exact Neural Network Inference Inside the Weights of Qwen Concept 27B", author: "Concept Country")
#set page(paper: "us-letter", margin: (x: 1in, y: 0.9in), numbering: "1",
  header: context { if counter(page).get().first() > 1 [#set text(8pt, fill: luma(110)); Exact Neural Network Inference Inside the Weights of Qwen Concept 27B · Working paper v1 #h(1fr) Concept Country] })
#set text(font: ("Libertinus Serif", "New Computer Modern"), size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.")
#show heading.where(level: 1): it => block(above: 1.3em, below: 0.7em, text(12pt, weight: "bold", it))
#show heading.where(level: 2): it => block(above: 1em, below: 0.5em, text(10.5pt, weight: "bold", style: "italic", it))
#show figure.caption: set text(9pt)
#show link: set text(fill: rgb("#1f4f6b"))
#let fig(path, cap, w: 100%) = figure(placement: auto, image(path, width: w), caption: cap)

#align(center)[
  #block(width: 86%)[
    // Titles are display text, not body copy: justification stretches the word
    // spacing badly at 16pt across a full measure.
    #set par(justify: false, leading: 0.45em)
    #text(16pt, weight: "bold")[Exact Neural Network Inference Inside the Weights of Qwen Concept 27B]
  ]
  #v(0.15em)
  #text(11pt)[An #isa processor compiled into #layerschip transformer layers, verified against a reference emulator]
  #v(0.6em)
  Concept Country \
  #text(9.5pt)[New York, NY · Working paper v1, 8 November 2026]
]

#v(0.6em)
#block(inset: (x: 2.2em))[
#set text(9.5pt)
*Abstract.* A language model can contain a working computer: not a learned approximation
of one, but an exact instruction-set implementation whose every state transition is
verified against a reference emulator. We port #upproject, by #upauthor, from
#upbase to a 27B hybrid-attention reasoning model, compiling an #isa machine with a
bignum coprocessor into #layerschip transformer layers placed in front of
#hostmodel. The machine's memory is the Gated DeltaNet recurrent state:
#romkb#[ KB] of code ROM, #rodatakb#[ KB] of rodata and #ramkb#[ KB] of RAM live in a
linear-attention cache. In standalone free-running verification, #totalinstrm#[ M]
instructions across #totalprogs programs produce exactly the reference token stream, and a
single end-to-end sweep on the merged checkpoint runs #swprogs programs and
#swinstrm#[ M] instructions with #swexact#[/]#swprogs matching the reference emulator and
#swdiverged divergences. We then compile two neural networks to this instruction set. An MNIST classifier's
forward pass runs inside the weights and matches the reference emulator exactly over
#mlpinstrm#[ M] instructions. A character-level GPT with attention, softmax and a residual
stream reproduces its integer specification as a compiled program, and on the merged
checkpoint its #gptfourinstrm#[ M] instructions also match the reference emulator
exactly. Both are
integer end to end because the machine has no FPU, a design the measurements force rather
than suggest: an #raw("expf") costs #opexpfrel int8 multiply-accumulates, so a float
softmax in the inner model would cost #fsratio#[×] every matrix multiplication it
contains. The host model also calls the machine through
its own native tool-call syntax: asked for a product, it emits a `calc` tool call, the
in-weights machine computes #calcgrouped and writes the tool response, and the model reads
its own answer back. No tool runs outside the weights. The host model's parameters survive
the merge intact: #shaidentical of #shaidentical tensors byte-identical by SHA-256 after
un-padding, with #shanormresc RMSNorm gains matching a documented rescale exactly and
#shapadnonzero unexpected nonzero padding entries. We report prominently that
#critmet of #crittotal pre-specified acceptance criteria are met and one is not: strict
greedy text identity against the unmodified host holds on only #textidentical of #textn
prompts at #textnew tokens. The cause is identified (bf16 re-rounding of the rescaled norm
gains) and bounded: teacher-forced top-1 agreement is #tfagree#[/]#tfpositions against a
harness self-agreement floor of #tffloor#[/]#tfpositions, so the discrepancy sits at the
measurement noise floor rather than above it, but a criterion that says identical and
measures #textidentical#[/]#textn has not been met, and we do not reinterpret it.
]

= Introduction

A transformer layer is a sequence of linear maps with a nonlinearity between them. A digital
circuit is also a sequence of linear maps with a nonlinearity between them, if you are
willing to call a lookup table a nonlinearity and a one-hot vector a wire. The observation
is old and usually left as an observation, because the interesting direction, compiling a
real instruction set into real weight matrices and then checking every state transition
against a reference, is a great deal of work for a result that looks like a stunt.

#upauthor did that work. #upproject compiles an #isa machine, with a bignum coprocessor and
a text bridge, into the weight matrices of #upbase. The circuit is exact: it is not trained
to imitate a CPU, it is constructed so that its forward pass *is* the CPU's state
transition, and its outputs are compared token-for-token against a reference emulator. The
machine's memory is the host architecture's own linear-attention state, which is the part
that makes the construction fit a transformer rather than fight it.

This paper reports a port, and the port is the contribution. We take that machine to a 27B
hybrid reasoning model and merge it into an *edited* checkpoint: #hostmodel, the subject of
our companion paper (#companion, #link("/research/weight-space-programming")[Bennington
2026, Concept Research]), which is itself Qwen3.8-27B carrying a rank-one weight-space
edit. The question the port answers is not whether a computer can live in weights;
#upauthor established that. It is whether the construction survives a 5× scale increase,
a different layer-type mix, and a host model whose weights have already been modified for
unrelated reasons, and whether the host survives having a computer bolted to its front.

Three results follow. The machine is exact at the new geometry, over
#totalinstrm#[ M] instructions. The host's weights are preserved bit-for-bit, which we
verify per tensor rather than assert. And the host's *behaviour* is preserved only to within
bf16 rounding, which is close enough to be invisible in aggregate and not close enough to
satisfy a criterion written as "identical". We report that failure as a failure.

= Related work

Two lines of work meet here and neither quite contains the result.

The first is weight-space programming: obtaining a direction from a model's activations and
writing it into the weights so a behaviour is compiled in rather than applied at inference.
Our companion paper replicates that method on the host model used here, and the
distinction worth carrying forward is that such edits are *statistical*: a rank-one
additive term that shifts a distribution, validated by benchmark deltas against a
norm-matched random control. The edit changes what the model tends to do.

The second is the literature on transformer expressivity: constructions showing that
attention can implement pointer chasing, that chain-of-thought extends what a fixed-depth
model can compute, that specific architectures are Turing-complete given unbounded
precision and tape. These results are existence proofs. They establish that a weight
configuration exists; they do not produce one that runs `gcc -O2` output.

#upproject sits in neither place. It is not statistical; there is no distribution to shift,
only a state transition that is either exactly right or wrong, checked at every step. And it
is not an existence proof: the machine is compiled, instantiated in bf16, and executed. The
closest useful analogy is not machine learning at all but a soft-core processor: an FPGA
bitstream describes a CPU in reconfigurable logic, and this describes one in reconfigurable
linear algebra. The economics are terrible and the fidelity is exact, which is the opposite
of the usual trade in this field.

What makes the host architecture suitable is specific and worth stating. Gated DeltaNet is a
linear-attention mechanism carrying a matrix-valued recurrent state updated by a delta rule.
With the decay set to retain and the keys one-hot, that state stops being a compressed
summary of context and becomes addressable memory: head #sym.times key selects a slot, the
delta rule writes, a matching query reads. The machine does not simulate RAM in a cache; the
cache *is* the RAM, with the write semantics a store instruction needs. A pure
softmax-attention stack has no comparable primitive, and this is why the construction is
tied to hybrid architectures rather than being general.

= Method

== What is being merged into what

The host is #hostmodel, a #layersqwen#[-layer] hybrid model with hidden width #hidold,
#nlinearqwen Gated DeltaNet layers and #nfullqwen full-attention layers. Its provenance
matters for reading our results: it is Qwen3.8-27B plus the rank-one edit characterised in
the companion paper, and the merge source is that edited checkpoint (#raw(sourcemodel)), not
stock Qwen. Every claim below about preserved weights is a claim about preserving the
*edited* model.

The machine occupies #layerschip additional Gated DeltaNet layers in front of the host's
stack, giving #layerstotal layers total. The compiled circuit uses #irdims intermediate
dimensions, #circheads DeltaNet heads and #ffnunits MLP units, distributed across those
#circlayers layers with at most #maxunits units and #maxheads heads in any one.

== Widening without disturbing

The circuit needs residual dimensions the host does not have, so the hidden width grows from
#hidold to #hidnew. This is the step that could destroy the host, and the construction
avoids it as follows. Every original tensor is zero-padded to the new width and never reads
or writes a chip dimension. On chip tokens the host's own layers are muted through a padding
column: DeltaNet's decay, write and output-gate projections are driven to #[#(-1024)], and
full attention additionally receives an anti-sink key, refit on this host.

One class of tensor cannot be left alone. RMSNorm averages over the hidden dimension, so
widening changes its denominator. The #shanormresc full-residual norm gains are therefore
rescaled by

$ g' = (1 + g) dot sqrt(#hidold slash #hidnew) - 1 $

which leaves the host's normalized activations unchanged up to one bf16 rounding. That
rounding is the entire source of the behavioural discrepancy in #ref(<identity>). It is not
an implementation defect; it is the arithmetic consequence of expressing a wider residual
stream in a checkpoint format whose norms must span the full width.

== Verification strategy

Three independent checks, because the ways this can go wrong are independent.

*Weight preservation* is checked per tensor: un-pad each original tensor from the merged
checkpoint, hash it, compare to the host's bytes. Pass means byte identity, not approximate
agreement. The rescaled norms are checked against the formula rather than against the
original bytes, and the padding regions are checked for unexpected nonzeros.

*Machine exactness* is checked against a reference RV32 emulator at two levels. Teacher
forced, the circuit is fed the reference trace and every predicted token compared. Free
running, the circuit generates its own trace for thousands of instructions with no
correction, and the whole stream is compared. The second is the real test: teacher forcing
cannot detect an error that would compound.

*Host behaviour* is checked by greedy decoding #textn prompts that do not invoke the
machine, #textnew tokens each, on both the merged checkpoint and the unmodified host, and
comparing token sequences. Because that test is a near-tie detector rather than a behaviour
test, we additionally measure teacher-forced top-1 agreement and KL on the host's own
continuations, and measure the *harness's own floor* by running the host
against itself through two different code paths. Without that floor the agreement number is
uninterpretable.

== Reproducibility of the numbers in this paper

Every statistic printed here is generated. `emit_results.py` reads the #nartifacts
measurement artifacts and writes `results/results.json`; `emit_numbers.py` turns that into
the definitions this document interpolates. No figure in the prose is transcribed. The
artifacts are vendored with their SHA-256 digests, so a re-run against changed inputs is
detectable.

= Results

== The machine is exact

#figure(
  table(columns: 4, align: (left, right, right, left),
    stroke: none,
    table.hline(),
    table.header[Check][Programs][Instructions][Result],
    table.hline(),
    [Teacher-forced IR suite], [—], [—], [#tfpassed passed / #tffailed failed],
    [Text bridge, host tokenizer], [—], [—], [#bridgepassed passed],
    [Free-running fuzz], [#fuzzprogs], [#fuzzinstr], [exact],
    [Free-running C (`gcc -O2`)], [#elfprogs], [#elfinstr], [exact],
    table.hline(),
    [Total free-running], [#totalprogs], [#totalinstr], [all exact],
    table.hline(),
  ),
  caption: [Circuit exactness at the 27B geometry. The teacher-forced suite covers
  riscv-tests rv32ui / rv32um / rv32uzba, compiled C, faults, CSRs, and ALU / MUL / DIV /
  branch fuzzing. Free-running runs generate their own traces with no correction and are
  compared token-for-token against the reference emulator.],
)

#totalinstr instructions across #totalprogs programs produce exactly the reference token
stream. The teacher-forced suite passes #tfpassed of #tfpassed checks. The upstream
reproduction at #upbase, run first as a control on our hardware, gave #upassed passed /
#ufailed failed / #uskipped skipped; the single failure was an upstream bug in the
free-running executor (#raw(ufailkind) in #raw(ufailtest)), fixed by a two-line patch, after which
the suite is green. We report it because it is the one place our numbers required changing
someone else's code.

=== A single end-to-end sweep

The figures above come from the build's phase-2 log. To put the exactness claim on evidence
the reader can recompute, we then ran one corpus end to end on the merged checkpoint and
retained a JSON record per program. Every number in this subsection is derived from those
#swprogs records by #raw("emit_results.py"), not transcribed.

#figure(
  table(columns: 4, align: (left, right, right, left),
    stroke: none,
    table.hline(),
    table.header[Check][Programs][Instructions][Result],
    table.hline(),
    [riscv-tests ISA], [#swisa], [—], [#swisaexact#[/]#swisa exact],
    [C programs (`gcc -O2`)], [#swcprogs], [—], [#swcexact#[/]#swcprogs exact],
    table.hline(),
    [Sweep total], [#swprogs], [#swinstr], [#swexact#[/]#swprogs exact],
    table.hline(),
  ),
  caption: [The RV32IM sweep on the merged checkpoint. #swdiverged programs diverged,
  #swerrored errored and #swunhalted failed to halt; the longest single program,
  #raw(swlongest), executes #swlongestinstr instructions.],
)

The sweep is graded on the circuit's own token stream. A merged bridge checkpoint echoes
every printed byte back as a host text token by design, so the stream a reader observes
contains tokens the reference trace does not, and raw token equality fails for any program
that prints: it holds for #swrawexact of #swprogs programs, against #swexact of #swprogs on
the circuit's stream with echoes removed.

Three properties of the records establish that the excess tokens are echoes rather than
drift. The partition is exact: of #swprogs programs, the #swprinting that print are exactly
the ones failing raw equality (#swprintingraw#[/]#swprinting), and all #swnonprinting
non-printing programs pass it (#swnonprintingraw#[/]#swnonprinting). Token excess equals
echo count program by program, with no exceptions across the corpus. And every echo is the
host text token of the byte the chip had just printed, giving #swechofail integrity failures
across #swechoes echoes.

The distinction between the circuit's stream and the stream a reader observes is a property
of in-weights machines with a text bridge generally, not of this build.

== The host model calls the machine and reads its own answer

Asked #[#emph[Use calc: what is 123456789 \* 987654321?]], the merged checkpoint emits its
native tool-call syntax for `calc`, the in-weights machine executes the multiplication and
writes #raw("<tool_response>" + calcanswer + "</tool_response>"), and the host then answers
#calcgrouped in prose. The exchange costs #calctexttok text tokens and #calcchiptok machine
tokens. Replaying the generation and checking every circuit-owned position gives
#calcwrong wrong predictions and #calcownermismatch ownership mismatches across
#calcowned owned positions out of #calcpositions total.

Nothing in this runs outside the weights. There is no interpreter, no sandbox, no function
dispatch to a host process: the tool call and the tool response are both produced by matrix
multiplications in the same forward pass.

#figure(
  table(columns: 6, align: (left, right, right, right, left, right),
    stroke: none,
    table.hline(),
    table.header[Program][Instrs][Tokens][Ref tokens][Machine stream][tok/s],
    table.hline(),
    [`crc32`], [#ccrc32instr], [#ccrc32toks], [#ccrc32reftoks], [exact], [#ccrc32toksec],
    [`strops`], [#cstropsinstr], [#cstropstoks], [#cstropsreftoks], [exact], [#cstropstoksec],
    [`mini-gmp` multiply], [#cgmpmulinstr], [#cgmpmultoks], [#cgmpmulreftoks], [exact#super[†]], [#cgmpmultoksec],
    table.hline(),
  ),
  caption: [C programs run free-running through the merged checkpoint's own stock
  HuggingFace chip modules in bf16. The machine instruction stream is exact for all three.
  #super[†] `mini-gmp`'s machine stream is exact and its printed product correct, but the
  run emits #cgmpmultokdelta tokens against the reference total, so its end-to-end
  exactness flag is false. #cprogsnexact of #cprogsn programs are
  token-exact end to end; #cprogsn of #cprogsn have exact machine streams.],
)

`mini-gmp` computes a 20-digit by 20-digit bignum multiplication in
#cgmpmulinstr instructions and prints #gmpoutput, with each printed byte echoed back as a
host text token, #cgmpmulechoes of #cgmpmulechoes correct. The run emits #cgmpmultoks
tokens against #cgmpmulreftoks reference tokens, first differing at position
#cgmpmulfirstmismatch, where an extra text echo appears in the tail. The machine stream is
exact and the printed product is correct, so the computation is right and the difference is
in how the tail of the text bridge is emitted.

== Neural network inference inside the machine <inference>

The sweep establishes that the machine runs the corpus it was built against. A stronger
question is whether it runs programs its construction did not anticipate, with access
patterns unlike anything in that corpus. We compiled two neural networks to #isa and ran
their forward passes on the merged checkpoint: a neural network computed by a CPU that is
itself a neural network's forward pass.

Both inner models are integer end to end, and that is a measurement rather than a
preference. The ISA has no FPU, so every float operation becomes a soft-float libgcc call.
We measured the cost on the machine itself with a micro-benchmark whose baseline loop is
subtracted, counting instructions on the reference emulator over #opiters iterations:

#figure(
  table(columns: 3, align: (left, right, right),
    stroke: none,
    table.hline(),
    table.header[Operation][Instructions][Relative],
    table.hline(),
    [int8 multiply-accumulate], [#opint8mac], [#opint8macrel#[×]],
    [float32 divide], [#opfloatdiv], [#opfloatdivrel#[×]],
    [float32 multiply-accumulate], [#opfloatmac], [#opfloatmacrel#[×]],
    [#raw("sqrtf")], [#opsqrtf], [#opsqrtfrel#[×]],
    [#raw("expf")], [#opexpf], [#opexpfrel#[×]],
    table.hline(),
  ),
  caption: [Instruction cost per operation on the in-weights machine, measured rather than
  assumed. An #raw("expf") costs #opexpfrel int8 multiply-accumulates.],
)

These numbers decide the design. A faithful float softmax in the inner transformer below
would issue #fsexps #raw("expf") calls per token at full context, costing #fsinstr
instructions against #fsmatmul for every matrix multiplication in the model combined: a
ratio of #fsratio. The nonlinearity would cost more than the model. So the inner models
use int8 weights with int32 and int64 accumulators, requantization by a calibrated
fixed-point multiplier (#raw("(acc * M0) >> shift"), two instructions given RV32M's
#raw("MULH")), RMSNorm with an exact integer floor-sqrt, and softmax by a 256-entry
exponential lookup table. Weights are #raw("const") and live in rodata, so the scarce
#inframkb#[ KB] of RAM holds only activations.

=== A classifier

The first inner model is an MNIST classifier, #mlparch, with #mlpparams int8 parameters
occupying #mlprodatakb#[ KB] of rodata. Quantization costs #mlpaccdelta percentage points
of test accuracy: #mlpfloatacc#[%] in float32, #mlpintacc#[%] in int8 over the full test
set. The program classifies #mlpbatch images and prints the winning logit alongside each
prediction, which makes the output sensitive to every accumulator bit rather than letting
a wrong multiply hide behind a still-correct argmax.

On the merged checkpoint the machine executes #mlpinstrm#[ M] instructions and the
machine's token stream matches the reference emulator exactly (#mlpexact), with
#mlpechoes echoes all matching their printed bytes. All #mlpbatchcorrect of #mlpbatch
predictions are correct and every printed logit is identical to the PyTorch reference.

=== A transformer inside a transformer

The second inner model is a character-level GPT: #gptarch, trained on #gptcorpus, with
#gptparams int8 parameters in #gptrodatakb#[ KB] of rodata. It has the same architectural
ingredients as its host, attention, softmax, a residual stream and autoregressive decode,
and it is compiled for
the machine the host's weights implement.

Quantization is calibrated, not guessed. #gptncalib activation tensors are profiled at
their 99.9th percentile over real batches and #gptnmul fixed-point multipliers derived from
the observed ranges. Guessing shift constants instead gave an integer validation loss of
27.6 against the float model's #gptfloatloss. With calibration the integer model reaches
#gptintloss, slightly better than float; we did not investigate why. Greedy continuation of #raw(gptprompt) agrees between the float and integer
models for #gptsharedprefix characters:

#align(center)[#block(width: 90%)[
  #set text(9pt)
  #table(columns: 2, align: (left, left), stroke: none,
    [float32], [#raw(gptfloatgen)],
    [int8], [#raw(gptspecgen)],
  )
]]

after which one tie-break differs and greedy decode amplifies it.

The expected token stream is derived from the emitted weight header rather than from the
training process, and the distinction is not pedantic. The trainer holds float scales and
calibration tables in memory; only the int8 arrays and the multiplier pairs ship. While the
specification came from the trainer's in-process path, the compiled program disagreed with
it at the second generated token even though a stage-by-stage trace showed the two agreeing
bit for bit on the first decode step. Re-deriving the specification from the header, using
exactly the arithmetic the C performs, is what made them agree.

Each generation step replays the whole prefix, so the instruction count grows
quadratically with the number of generated tokens. Compiled for #isa, the program
reproduces its specification token for token under QEMU.
#if gptfourexact == "pending" [
  Its run on the merged checkpoint was still in progress when this version was built, so we
  report no checkpoint result for it: the classifier above is the verified demonstration,
  and the transformer is so far verified only as a program, not yet as a program executed
  by the weights.
] else [
  On the merged checkpoint a #[4]-token generation executes #gptfourinstrm#[ M]
  instructions through the Triton executor, and the machine's token stream against the
  reference emulator is graded exact: #gptfourexact.
]
#if mlpexecexact != "pending" [
  The executor is a different path from the stock HuggingFace chip modules the sweep used,
  so we ran the MNIST program through both: they agree on instruction count, output and
  verdict (#mlppathsagree), with the executor #execspeedup#[×] faster.
]

#figure(
  table(columns: 5, align: (left, right, right, left, right),
    stroke: none,
    table.hline(),
    table.header[Inner model][Params][Instructions][Machine stream][tok/s],
    table.hline(),
    [MNIST MLP], [#mlpparams], [#mlpinstr], [#mlpexact], [#mlptoksec],
    ..if mlpexecexact != "pending" { ([MNIST MLP (executor)], [#mlpparams], [#mlpinstr], [#mlpexecexact], [#mlpexectoksec]) },
    ..if gptfourexact != "pending" { ([Character GPT, 4 tokens], [#gptparams], [#gptfourinstr], [#gptfourexact], [#gptfourtoksec]) },
    table.hline(),
  ),
  caption: [Neural network inference on the merged checkpoint. The grading metric is the
  machine's own token stream against the reference emulator, with printed-byte echoes
  checked and excluded as in the sweep.],
)

=== Quantizing the chip layers breaks the machine

A text model degrades gracefully under quantization. The chip layers are not a text model,
so we tested them separately.
#if qpstate == "done" [
  We rounded the #qptensors chip-layer weight tensors that a standard recipe would quantize
  onto the weight grids of #qplossy lossy formats (FP8 block, and llama.cpp's 8-bit and
  4-bit block formats), leaving the #qpexcluded tensors such recipes keep in bf16
  untouched, and ran #qpprogs programs after each. The bf16 round-trip control stays exact
  on all of them (#qpcontrol). Of the #qplossy lossy grids, #qplossyexact keep the machine
  exact, and #qplossyprogsexact program runs in total survive. The gentlest grid that breaks it perturbs a weight by at most
  #qpsmallesterr#[%] of its tensor's largest magnitude. This simulates weight grids only,
  not a quantized inference kernel, so it shows that these formats cannot be applied to the
  chip layers, and says nothing yet about a checkpoint that quantizes only the host's
  layers and keeps the chip layers in bf16.
]

What this shows is that the processor in the weights is a general computer: it runs
a kind of program its test corpus never exercised, and the classifier's result is
bit-exact against an independent emulator. What it does not show is any benefit to the
host. A conventional CPU performs this inference in microseconds. The result is
computational universality, not performance.

== One criterion is not met <identity>

The build was specified against seven acceptance criteria. #critmet are met. The one that is
not is text identity against the unmodified host.

#figure(
  table(columns: 2, align: (left, left),
    stroke: none,
    table.hline(),
    table.header[Criterion][Result],
    table.hline(),
    ..criteriarows.map(r => (r.at(0), [#r.at(1) — #text(9pt)[#r.at(2)]])).flatten(),
    table.hline(),
  ),
  caption: [All seven pre-specified acceptance criteria, generated from the results file so
  the verdicts cannot drift from the measurements.],
)

Greedy decoding #textn prompts for #textnew tokens gives #textidentical identical
continuations and #textdivergent divergent ones (prompts #textprompts), first diverging at
token positions #textpositions. Both versions stay fluent and on topic where they differ.
The criterion said identical on #textn prompts. #textidentical is not #textn.

What we can establish is the size of the effect. Teacher-forced on the host's own
continuations across #tfpositions positions, top-1 agreement is #tfagree#[/]#tfpositions
with mean KL #tfklmean and max #tfklmax. Running the *host against itself* through the two
code paths the harness uses, prefill versus incremental decode, gives
#tffloor#[/]#tfpositions. The merged checkpoint therefore agrees with the host
#tfabovefloor positions better than the host agrees with itself under a change of kernel
path. The worst single prompt (#tfworstprompt) scores #tfworstagree#[/]#textnew, and it
scores #tfworstagree#[/]#textnew on the self-comparison too.

The honest reading is that the discrepancy is at or below the measurement floor, and that
the floor is not zero. Greedy decoding is a near-tie amplifier: a logit shift of about one
ulp flips an argmax wherever two tokens are close, and the flip then propagates. The
upstream project reports the same picture at #upbase, KL #tfupstreamkl against a
#tfupstreamfloor floor.

This does not rescue the criterion. Exact identity would require RMSNorm that ignores the
padding dimensions, which the stock checkpoint format cannot express; the alternative is to
restate the criterion as a KL bound, which is what upstream does and which we decline to do
retroactively. The criterion as written is not met, and the correct summary of this build is
#critmet of #crittotal.

== Throughput

#figure(
  table(columns: 3, align: (left, right, left),
    stroke: none,
    table.hline(),
    table.header[Path][Trace tok/s][Note],
    table.hline(),
    [Triton executor, batch 1], [#spexecb], [#spexecinstr instr/s],
    [Triton executor, batch 8], [#spexecbeight], [aggregate],
    [Triton executor, batch 32], [#spexecbthirtytwo], [#spexecinstrthirtytwo instr/s aggregate],
    [`torch.compile` executor, batch 1], [#sptorchcompile], [same circuit],
    [Stock HF chip modules, batch 1], [#sphfcrc], [no custom runtime],
    [#upbase in chat, executor], [#spchatfourb], [end to end, both models resident],
    [27B in chat, host streamed from disk], [#spchattwentyseven], [I/O bound — not representative],
    table.hline(),
  ),
  caption: [Measured on one #host. All executor rows verified exact. The batch-1 executor is
  #spexecoverhf#[#sym.times] the stock-module path; batching to 32 gives
  #spbatchspeedup#[#sym.times] aggregate throughput over batch 1.],
)

The last row needs disclaiming rather than quoting. A full-speed 27B chat run needs about
#spmemneeded#[ GB] of accelerator memory, which was not available while production work
occupied the host; the model's own layers were streamed from disk, and
#spchattwentyseven tok/s measures that I/O path, not the model. The single `calc` exchange
took #spchatseconds minutes for this reason. We report it for completeness and attach no
meaning to it. The honest statement about 27B chat throughput is that it is unmeasured.

== Where the parameters go

#figure(
  table(columns: 2, align: (left, right),
    stroke: none,
    table.hline(),
    table.header[Component][Params],
    table.hline(),
    [#hostmodel at native width #hidold], [#pqwennative B],
    [#sym.plus zero-padding those #layersqwen layers to #hidnew], [#ppadding B],
    [#sym.plus #layerschip chip layers (the machine)], [#pchiplayers B],
    [#sym.plus machine vocabulary rows (embedding and head)], [#pchipvocab B],
    table.hline(),
    [merged checkpoint], [#ptotal B],
    table.hline(),
  ),
  caption: [Parameter decomposition, computed from the actual layer-type sequence with
  Gated DeltaNet and gated-attention shapes. The components sum to the total
  (closure: #paramcloses). Model validated against the checkpoint's known tensor count
  (#ntensorspred predicted vs #ntensors reported: #tensorvalidated).],
)

The headline growth from 27B to #ptotal B is mostly not information. #ppadding B,
#ppaddingpct#[%] of the checkpoint, is zeros, present only because a wider residual stream
must be expressed in a format that stores every tensor at full width. The machine itself is
#pchiplayers B of live parameters plus #pchipvocab B of vocabulary rows for its trace
tokens.

#nlinear of the #layerstotal layers are Gated DeltaNet, whose projections are shaped
differently from standard multi-head attention, and `attn_output_gate` is true, which fuses
a sigmoid gate into `q_proj` and doubles its output width. The gate shares the `q_proj`
tensor, so it contributes #pgatecost#[ B] parameters without changing the tensor count:
a decomposition without it gives #pnogate#[ B] against the measured #ptotal#[ B].

The hidden size grows from #hidold to #hidnew. Of the added dimensions, #ballast are
ballast, 1 is tau and #slots are chip slots, #widthstated in all; the remaining
#widthresidual bring #hidnew to #headmultnew#[#sym.times] the head dimension. The vocabulary is
#vocabbase host rows plus #vocabchip machine rows, #vocabstated in all, stored as #vocab,
a multiple of 64.

The checkpoint is #pgbdec#[ GB] (#pgib#[ GiB]) in bf16.

It loads as a stock #raw(arch) with #ntensors tensors, #shamissing missing,
#shamismatch shape mismatches and #shamismatch unexpected keys. No custom runtime is
required for correctness. The custom decoder exists only for speed: it lets machine tokens
skip the host's #layersqwen layers, which plain `generate()` would otherwise run for every
one of the #totaltoks trace tokens.

== The host's weights survive

#figure(
  table(columns: 2, align: (left, right),
    stroke: none,
    table.hline(),
    table.header[Check][Count],
    table.hline(),
    [Tensors byte-identical after un-padding (SHA-256)], [#shaidentical / #shaidentical],
    [RMSNorm gains matching the documented rescale exactly], [#shanormresc / #shanormresc],
    [Mismatches], [#shamismatch],
    [Missing tensors], [#shamissing],
    [Padding entries unexpectedly nonzero], [#shapadnonzero],
    [Vision / MTP tensors dropped (text-only build)], [#shadropped],
    table.hline(),
  ),
  caption: [Per-tensor preservation of the host model. #shachecked tensors checked.
  Byte identity, not approximate agreement.],
)

Every one of the host's #shaidentical carried tensors is byte-identical to the source
checkpoint, and every one of the #shanormresc tensors that had to change equals the
documented transform of the source bytes exactly. The rank-one edit from the companion paper
is carried through untouched, because it lives in those bytes.

= Discussion

The narrow claim is secure and worth stating precisely. An exact #isa machine, with
#romkb#[ KB] of code ROM and #ramkb#[ KB] of RAM held in linear-attention state, runs inside
the weight matrices of a 27B reasoning model, executes #totalinstrm#[ M] instructions
identically to a reference emulator across the phase-2 suite and a further #swinstrm#[ M]
across the #swprogs#[-program] end-to-end sweep, and is invoked by the host model through
the host's own tool-call syntax with no external interpreter. The host's parameters are
preserved byte-for-byte. This is a port of #upauthor's construction and the construction is
his; what we establish is that it survives the scale change, the different layer mix, and an
already-edited host.

The interesting consequence is about what a tool call can be. A model that calls an external
calculator depends on a runtime contract: the tool exists, is reachable, returns what it
claims. Here the contract is the checkpoint. The arithmetic is as portable as the weights,
and it cannot be removed without removing parameters. Whether that is useful is a separate
question: #sphfcrc tok/s through stock modules for a machine that a CPU would run billions
of times faster is not a performance argument. It is an argument about where a capability
*lives*.

The contrast with the companion paper is the cleanest way to see what is unusual. That work
writes a direction into nine matrices and shifts a distribution; it is validated
statistically, against a norm-matched control, because the effect is statistical. This work
writes a state machine into eleven layers; it is validated by exact comparison, because
there is nothing to average. Both are weight-space programming. One compiles a tendency and
the other compiles a program, and the second needs no control arm because an emulator
disagreement is not a matter of degree.

What this does not establish is that the approach generalises beyond hybrid
linear-attention architectures. The construction depends on Gated DeltaNet state being
addressable memory. It does not establish any benefit to the host model: we measured that
the host is unharmed, not that it is improved, and the only tool wired to the machine is
`calc`. And it does not establish throughput at 27B in a realistic serving configuration,
because we could not fit the model.

= Limitations

The failed criterion is the first limitation and we have stated it plainly: greedy text is
not bit-identical to the host, #textidentical of #textn prompts at #textnew tokens, caused
by bf16 re-rounding of the rescaled norm gains. The KL evidence bounds the effect at the
measurement floor; it does not satisfy the criterion. Exact identity needs norms that ignore
padding dimensions, which the stock format cannot express.

Full-GPU throughput for the 27B merge is unmeasured. The figure in our speed table for 27B
chat (#spchattwentyseven tok/s) measures disk streaming on a host with insufficient free
memory and should not be cited as a throughput result. The machine needs about
#spmemneeded#[ GB] to run resident, and the host that would have provided it was down at
the time of writing and remains so, which is also why no further measurement was attempted
for this draft.

The text-identity evaluation is #textn prompts at #textnew tokens, which is a small sample
for a near-tie detector; the teacher-forced comparison is #tfpositions positions on the
host's own continuations and therefore measures agreement on a distribution the host
generated, not on held-out text. We did not evaluate the merged checkpoint on any capability
benchmark, so the claim "the host is unharmed" rests on token agreement and weight identity,
not on task performance. A reader who wants to know whether #hostmodel's benchmark results
from the companion paper survive the merge should treat that as untested.

The machine's correctness is established against one reference emulator. A bug shared
between the compiler and the reference would be invisible to every check here. The
riscv-tests suites constrain this but do not eliminate it.

Numerics are a standing hazard: the chip layers require exact bf16 matmul accumulation
(`allow_bf16_reduced_precision_reduction = False`), and a default-configured runtime will
produce wrong results silently. The bundled scripts set it; a third party reproducing this
must.

The build is text-only (the host's vision tower is not carried over) and the host
checkpoint itself is not final pending its own evaluation verdict, so the merge may be
rebuilt. #hostmodel RV32IM 27B is experimental and under evaluation. Nothing here is a
release.

Finally, the one-paragraph caveat on the whole exercise: we are reporting a port of someone
else's construction, measured on our own hardware with our own host model. The parts that
would be most valuable to a reader, why the circuit is correct and how the compiler lowers
RV32 semantics into attention, are documented in #upproject, not here.

#v(0.4em)
*Status.* Working paper v1. #critmet of #crittotal pre-specified acceptance criteria met;
#critfailed not met (strict greedy text identity, #textidentical#[/]#textn). The model,
#hostmodel RV32IM 27B, is experimental and under evaluation.

*Model availability.* The merged model is published as `conceptcountry/concept-rv32im-27b`
on #link("https://models.concept.country")[models.concept.country] and on
#link("https://huggingface.co/conceptcountry")[HuggingFace], in bf16 and FP8. FP8 is what
brings it within a single 128 GB accelerator; bf16 is what every measurement here used.
Both carry the vision tower and the multi-token-prediction head, which an earlier merge
dropped.

*Data and code.* The #nartifacts measurement artifacts are vendored with SHA-256 digests
alongside `emit_results.py`, which recomputes every statistic, and `emit_numbers.py`, which
generates the definitions this document interpolates, at
#link("https://github.com/conceptcountry/concept-rv32im")[github.com/conceptcountry/concept-rv32im],
Apache-2.0 as the upstream work requires. No number in this text is transcribed
by hand.

*Acknowledgements.* #upproject is the work of *#upauthor*
(#link("https://github.com/divagr18/qwenputer")[#upurl]): code #upcode, weights
#upweights. The circuit, the compiler, the RV32 firmware and the text bridge are his; its
internal architecture is called #upinternal. Our contribution is fitting the compiler to the
27B geometry, merging into #hostmodel, and the verification reported here. Our model name
reuses neither #upproject nor #upinternal, so the credit is stated in prose instead: this is
a port, and the original is his. The base model is #basemodel (Qwen team, Apache-2.0). The
host edit and its evaluation are described in the companion paper
(#link("/research/weight-space-programming")[#companion]). The analysis, figures and text
are the authors'.
