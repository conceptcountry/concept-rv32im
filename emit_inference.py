#!/usr/bin/env python3
"""Derive the inference-in-the-weights figures from the run artifacts.

Nothing in the paper's inference section is transcribed by hand. This reads the result
JSONs written by c27.py / c27_exec.py, the training metadata written by train_*.py, and
the operation-cost measurement written by bench_ops.c, and emits one JSON that
emit_numbers.py turns into Typst definitions.

Inputs expected in artifacts/:
  c-mlp.json           MNIST MLP on the merged checkpoint, HF module path
  c-mlp-exec.json      the same program through the Triton executor (cross-path check)
  c-gpt4.json          character GPT, 4 generated tokens, executor path
  mlp_meta.json        MLP training and quantization metadata
  gpt_meta.json        GPT training, calibration and quantization metadata
  bench_ops.json       instruction cost per operation, measured on the reference emulator

A missing optional artifact is recorded as absent rather than guessed, so a partial run
cannot silently publish a figure it did not measure.
"""
import datetime
import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
OUT = os.path.join(HERE, "results", "inference.json")

REQUIRED = ["c-mlp.json", "mlp_meta.json", "gpt_meta.json", "bench_ops.json"]
OPTIONAL = ["c-mlp-exec.json", "c-gpt4.json", "c-gpt8.json", "quant_probe_chip.json"]


def art(name, required=True):
    path = os.path.join(ART, name)
    if not os.path.isfile(path):
        if required:
            raise SystemExit(f"missing required artifact: {name}")
        return None
    with open(path) as f:
        return json.load(f)


for name in REQUIRED:
    art(name)

mlp_hf = art("c-mlp.json")
assert mlp_hf is not None
mlp_ex = art("c-mlp-exec.json", required=False)
gpt4 = art("c-gpt4.json", required=False)
gpt8 = art("c-gpt8.json", required=False)
mlp_meta = art("mlp_meta.json")
assert mlp_meta is not None
gpt_meta = art("gpt_meta.json")
assert gpt_meta is not None
bench = art("bench_ops.json")
assert bench is not None

R: dict[str, object] = {"generated_utc": datetime.datetime.now(datetime.timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%SZ")}


def run_record(d):
    """The subset of a result JSON that is evidence, with the grading metric explicit."""
    if d is None:
        return None
    return {
        "program": d["program"],
        "exact_chip_stream": bool(d["exact_chip_stream"]),
        "exact_raw_including_echoes": bool(d["exact"]),
        "instructions": int(d["instructions"]),
        "tokens": int(d["tokens"]),
        "reference_tokens": int(d["reference_tokens"]),
        "echoes": int(d["echoes"]),
        "echoes_equal_printed_bytes": bool(d["echoes_equal_printed_bytes"]),
        "halted": bool(d["halted"]),
        "first_mismatch": d["first_mismatch"],
        "seconds": float(d["seconds"]),
        "tok_per_s": float(d["tok_per_s"]),
        "output": d["output"],
        "path": d["path"],
    }


R["mlp_run"] = run_record(mlp_hf)
R["mlp_run_executor"] = run_record(mlp_ex)
R["gpt4_run"] = run_record(gpt4)
R["gpt8_run"] = run_record(gpt8)

# The two paths must agree where both ran: the same program, the same instruction count,
# the same verdict. Disagreement would mean one path is not running what it claims.
if mlp_ex is not None:
    R["mlp_paths_agree"] = bool(
        mlp_hf["instructions"] == mlp_ex["instructions"]
        and mlp_hf["exact_chip_stream"] == mlp_ex["exact_chip_stream"]
        and mlp_hf["output"] == mlp_ex["output"])
    R["executor_speedup_over_hf"] = round(mlp_ex["tok_per_s"] / mlp_hf["tok_per_s"], 1)
else:
    R["mlp_paths_agree"] = None
    R["executor_speedup_over_hf"] = None

R["mlp_model"] = {
    "arch": mlp_meta["arch"],
    "params_int8": int(mlp_meta["params_int8"]),
    "rodata_bytes": int(mlp_meta["rodata_bytes"]),
    "float_accuracy": float(mlp_meta["float_test_acc"]),
    "int_accuracy": float(mlp_meta["int8_test_acc"]),
    "accuracy_delta": round(float(mlp_meta["float_test_acc"])
                            - float(mlp_meta["int8_test_acc"]), 4),
    "requant_shift": int(mlp_meta["shift1"]),
    "batch_images": len(mlp_meta["test_indices"]),
    "expected_predictions": [int(x) for x in mlp_meta["expected_pred"]],
    "expected_labels": [int(x) for x in mlp_meta["test_labels"]],
    "batch_correct": sum(int(a) == int(b) for a, b in
                         zip(mlp_meta["expected_pred"], mlp_meta["test_labels"])),
}

# Geometry is recorded in the arch string ("char GPT V=65 d=64 L=2 H=2 ff=256 ctx=64"),
# so it is parsed from the artifact rather than restated here.
_a = dict(re.findall(r"(\w+)=(\d+)", gpt_meta["arch"]))
R["gpt_model"] = {
    "arch": gpt_meta["arch"],
    "params_int8": int(gpt_meta["params_int8"]),
    "rodata_bytes": int(gpt_meta["rodata_bytes"]),
    "float_val_loss": float(gpt_meta["float_val_loss"]),
    "int_val_loss": float(gpt_meta["int_val_loss"]),
    "prompt": gpt_meta["prompt"],
    "float_generation": gpt_meta["float_generation"],
    "spec_generation": gpt_meta["spec_generation"],
    "vocab": int(_a["V"]),
    "layers": int(_a["L"]),
    "d_model": int(_a["d"]),
    "heads": int(_a["H"]),
    "d_ff": int(_a["ff"]),
    "ctx": int(_a["ctx"]),
    "n_calibrated_tensors": len(gpt_meta["calibration"]),
    "n_multipliers": len(gpt_meta["multipliers"]),
    "score_scale_nats": float(gpt_meta["score_scale_nats"]),
    "corpus": "tiny Shakespeare",
}

# How far the float and integer models agree before greedy decode amplifies a tie-break.
f, s = gpt_meta["float_generation"], gpt_meta["spec_generation"]
shared = 0
for a, b in zip(f, s):
    if a != b:
        break
    shared += 1
R["gpt_model"]["float_int_shared_prefix_chars"] = shared
R["gpt_model"]["int_beats_float"] = bool(
    float(gpt_meta["int_val_loss"]) < float(gpt_meta["float_val_loss"]))

ops = bench["ops"]
R["op_costs"] = {
    "baseline_instructions": int(bench["baseline_instructions"]),
    "iterations": int(bench["iterations"]),
    "per_op": {k: {"instructions": v["instructions_per_op"],
                   "relative_to_int8_mac": v["relative_to_int8_mac"]}
               for k, v in ops.items()},
}

# The cost a faithful float softmax would carry in this GPT, which is the measurement that
# rules out a direct llama2.c port rather than an assertion that it is too slow.
g = R["gpt_model"]
softmax_exps = g["layers"] * g["heads"] * g["ctx"]
R["float_softmax_cost"] = {
    "exps_per_token": softmax_exps,
    "instructions_per_token": round(softmax_exps * ops["expf"]["instructions_per_op"]),
    "note": "expf calls for one token at full context, at the measured soft-float cost",
}
macs_per_token = (g["layers"] * (4 * g["d_model"] ** 2 + 2 * g["d_ff"] * g["d_model"]
                                + g["heads"] * g["ctx"] * (g["d_model"] // g["heads"]) * 2)
                  + g["vocab"] * g["d_model"])
R["float_softmax_cost"]["matmul_instructions_per_token"] = round(
    macs_per_token * ops["int8_mac"]["instructions_per_op"])
R["float_softmax_cost"]["ratio_to_matmuls"] = round(
    R["float_softmax_cost"]["instructions_per_token"]
    / R["float_softmax_cost"]["matmul_instructions_per_token"], 2)

qp = art("quant_probe_chip.json", required=False)
if qp is None:
    R["quant_probe_chip"] = None
else:
    # The NVFP4 grid in this probe run cast group scales to fp8 without the format's global
    # scale, which zeroes small groups (max relative error exactly 1.0). That is a defect in
    # the simulation, not a property of NVFP4, so the row is not reported.
    fm = {k: v for k, v in qp["formats"].items() if k != "nvfp4a16"}
    R["quant_probe_chip"] = {
        "n_quantized_tensors": int(qp["n_quantized_tensors"]),
        "n_excluded_tensors": int(qp["n_excluded_tensors"]),
        "n_programs": len(qp["programs"]),
        "formats": {k: {"n_exact": int(v["n_exact"]), "n_programs": int(v["n_programs"]),
                        "max_relative_weight_error": float(v["max_relative_weight_error"])}
                    for k, v in fm.items()},
        "control_exact": bool(fm["bf16"]["n_exact"] == fm["bf16"]["n_programs"]),
        "n_lossy_formats": sum(1 for k in fm if k != "bf16"),
        "n_lossy_formats_exact": sum(1 for k, v in fm.items()
                                     if k != "bf16" and v["n_exact"] == v["n_programs"]),
        "lossy_programs_exact": sum(v["n_exact"] for k, v in fm.items() if k != "bf16"),
        "smallest_breaking_error": min(float(v["max_relative_weight_error"])
                                       for k, v in fm.items()
                                       if k != "bf16" and v["n_exact"] < v["n_programs"]),
    }

R["machine"] = {"code_rom_kb": 256, "rodata_kb": 128, "ram_kb": 32, "isa": "RV32IM + Zba"}

# The release formats. Read from the build's own verification output rather than
# restated, so a format whose check failed cannot be described here as having passed.
_verify = os.path.join(ART, "release-verify.json")
_text = os.path.join(ART, "release-text.json")
if os.path.isfile(_verify):
    with open(_verify) as fh:
        V = json.load(fh)
    T = {}
    if os.path.isfile(_text):
        with open(_text) as fh:
            T = json.load(fh)

    ORDER = ["bf16", "fp8", "nvfp4a16", "Q8_0", "Q4_K_M"]
    SCHEME = {
        "bf16": "bfloat16",
        "fp8": "FP8_BLOCK, 128x128 blocks",
        "nvfp4a16": "NVFP4 weights, group 16, bf16 activations",
        "Q8_0": "GGUF Q8_0",
        "Q4_K_M": "GGUF Q4_K_M, from the bf16 GGUF",
    }
    rows = []
    for key in ORDER:
        v = V["formats"].get(key)
        if not v:
            continue
        t = (T.get(key) or {})
        is_gguf = v.get("kind") == "gguf"
        rows.append({
            "format": key,
            "scheme": SCHEME[key],
            "machine_unchanged": bool(v.get("machine_unchanged")),
            "machine_basis": "dtype" if is_gguf else "sha256",
            "n_machine_tensors": v.get("n_machine_tensors"),
            "n_machine_identical": (v.get("n_machine_tensors") if is_gguf
                                    else v.get("identical")),
            "host_quantized_tensors": (v.get("host_quantized_tensors")
                                       if is_gguf else v.get("n_compressed_host_tensors")),
            "loads": v.get("loads"),
            "text_identical": t.get("n_identical"),
            "text_prompts": t.get("n_prompts"),
            "text_measured": t.get("n_identical") is not None,
            "text_median_first_divergence_char": t.get("median_first_divergence_char"),
        })

    R["release"] = {
        "formats": rows,
        "n_formats": len(rows),
        "n_formats_machine_unchanged": sum(1 for r in rows if r["machine_unchanged"]),
        "n_formats_load": sum(1 for r in rows if r["loads"]),
        "n_formats_text_measured": sum(1 for r in rows if r["text_measured"]),
        "all_machine_unchanged": all(r["machine_unchanged"] for r in rows),
        "all_load": all(r["loads"] for r in rows),
        "text_method": (T.get("method") or {}),
        "machine_layers": 11,
        "repo": "ConceptCountry/concept-rv32im-27b",
    }

# The release build against the build the results were measured on.
_cmp = os.path.join(ART, "cmp-builds.json")
if os.path.isfile(_cmp):
    with open(_cmp) as fh:
        C = json.load(fh)
    g = C["groups"]
    R["release_build"] = {
        "tensors_measured_build": C["tensors_a"],
        "tensors_release_build": C["tensors_b"],
        "vision_tensors": C["only_in_b"].get("visual"),
        "mtp_tensors": C["only_in_b"].get("mtp"),
        "only_in_measured_build": C["only_in_a"],
        "machine_tensors": g["chip_layers"]["n"],
        "machine_identical": g["chip_layers"]["identical"],
        "machine_dtype_differs": g["chip_layers"]["dtype_differs"],
        "host_tensors": g["host_layers"]["n"],
        "host_identical": g["host_layers"]["identical"],
        "norms_float32_instead_of_bf16": (g["host_layers"]["dtype_differs"]
                                          + g["embed_head_norm"]["dtype_differs"]),
    }

R["artifact_digests"] = {}
for name in sorted(os.listdir(ART)):
    path = os.path.join(ART, name)
    if not os.path.isfile(path):
        continue
    with open(path, "rb") as fh:
        R["artifact_digests"][name] = hashlib.sha256(fh.read()).hexdigest()
R["n_artifacts"] = len(R["artifact_digests"])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(R, fh, indent=2)
    fh.write("\n")

print(f"wrote {OUT}")
print(f"  MLP  exact_chip_stream={R['mlp_run']['exact_chip_stream']} "
      f"over {R['mlp_run']['instructions']:,} instructions")
if R["gpt4_run"]:
    print(f"  GPT4 exact_chip_stream={R['gpt4_run']['exact_chip_stream']} "
          f"over {R['gpt4_run']['instructions']:,} instructions")
else:
    print("  GPT4 not yet present (run still in flight)")
print(f"  float softmax would cost {R['float_softmax_cost']['ratio_to_matmuls']}x "
      f"the model's matmuls")
