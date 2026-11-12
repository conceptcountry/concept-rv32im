#!/usr/bin/env bash
# Greedy text accuracy, each release format against the same format of the base model.
#
# Streams layers from disk (stream_dq dequantizes compressed shards on read), so peak
# residency is a couple of layers rather than a whole checkpoint. QWP_RESIDENT=0 because
# the compact-resident path needs the chipformer env, which does not carry
# compressed-tensors; nothing here executes the machine, it only generates text.
set -u
D=/home/somewheresystems/qwenputer/dist
BASE=/home/somewheresystems/wsp/dist/qwen-concept-27b
cd ~/qwenputer
export QWP_RESIDENT=0
PY=~/wsp/venv/bin/python
N=${N_NEW:-48}

run() {  # name dir prompts text_vocab
  local name=$1 dir=$2 prompts=$3 tv=$4
  if [ -f "$D/gen_$name.json" ]; then echo "##### $name already done"; return 0; fi
  if [ ! -d "$dir" ]; then echo "##### $name SKIP: $dir missing"; return 0; fi
  echo "##### $name  ($dir)  text_vocab=$tv"
  timeout 5400 "$PY" rv_text_eval.py --model "$dir" --prompts "$D/$prompts" \
    --n-new "$N" --text-vocab "$tv" --out "$D/gen_$name.json" 2>&1 | tail -2
}

# The merged build's vocabulary carries machine tokens above the text range, so its
# argmax is taken over the text vocabulary only. The base model's vocabulary is all text.
run rv-bf16      "$D/bf16"      prompts_rv.json   248320
run base-bf16    "$BASE/bf16"   prompts_base.json 0
run rv-fp8       "$D/fp8"       prompts_rv.json   248320
run base-fp8     "$BASE/fp8"    prompts_base.json 0
run rv-nvfp4a16  "$D/nvfp4a16"  prompts_rv.json   248320
run base-nvfp4a16 "$BASE/nvfp4a16" prompts_base.json 0

echo "=== comparisons ==="
for f in bf16 fp8 nvfp4a16; do
  if [ -f "$D/gen_rv-$f.json" ] && [ -f "$D/gen_base-$f.json" ]; then
    echo "--- $f ---"
    "$PY" rv_text_eval.py --compare "$D/gen_rv-$f.json" "$D/gen_base-$f.json" \
      --out "$D/cmp_$f.json" 2>&1 | tail -2
  else
    echo "--- $f: not both generated, not measured ---"
  fi
done
echo "=== $(date -Is) TEXT_DONE ==="
