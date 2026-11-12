#!/usr/bin/env bash
# Rebuild the GGUF formats without the MTP head.
#
# Why: the first build exported the MTP (NextN) tensors, and llama.cpp refuses the result:
#
#   check_tensor_dims: tensor 'blk.75.nextn.enorm.weight' has wrong shape;
#   expected 6656, got 5120
#
# The merge widens the host from 5120 to 6656 and carries the MTP head along, but two of
# its 15 tensors (mtp.pre_fc_norm_embedding, mtp.pre_fc_norm_hidden) are norms over the
# *embedding* stream and keep the host's 5120 width, which is correct for the safetensors
# release and loads fine as the stock transformers class. llama.cpp's NextN loader derives
# every nextn norm's expected width from the model's hidden size, so a 5120 norm in a 6656
# model fails its dimension check.
#
# The safetensors formats keep the head. For GGUF it is excluded with --no-mtp, which is
# also what the original Qwen Concept 27B shipped: its own Q8_0 GGUF carries zero nextn
# tensors. The vision tower is unaffected and still ships as a separate mmproj file.
#
# Q4_K_M is quantized from the bf16 GGUF, never from the Q8_0.
set -u
Q=$HOME/qwenputer
OUT=$Q/dist
PY=$HOME/wsp/venv/bin/python
QZ=$HOME/llama.cpp/build/bin/llama-quantize
CLI=$HOME/llama.cpp/build/bin/llama-cli
CONV=$HOME/llama.cpp/convert_hf_to_gguf.py
N=concept-rv32im-27b
G=$OUT/gguf
TMP=$G/.tmp-bf16-nomtp.gguf

echo "=== $(date -Is) gguf rebuild (no MTP) ==="
df -h "$Q" | tail -1

# A partial intermediate from an interrupted or OOM-killed run is not resumable: the
# converter writes sequentially and does not checkpoint, so a short file would be
# silently quantized into a broken release. Only a file marked complete is reused.
if [ -f "$TMP" ] && [ ! -f "$TMP.complete" ]; then
  echo "  discarding incomplete intermediate ($(stat -c %s "$TMP") B)"
  rm -f "$TMP"
fi
if [ ! -f "$TMP" ]; then
  echo "=== bf16 gguf, --no-mtp ==="
  "$PY" "$CONV" "$OUT/bf16" --no-mtp --outtype bf16 --outfile "$TMP" 2>&1 | tail -12
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "  CONVERT FAILED rc=$rc"; rm -f "$TMP"
    echo "=== $(date -Is) REBUILD_FAILED ==="; exit 1
  fi
  touch "$TMP.complete"
fi
[ -f "$TMP" ] || { echo "CONVERT FAILED"; exit 1; }

cat > "$G/tensor-types.txt" <<EOF
^blk\.([0-9]|10)\..*=bf16
EOF

for q in Q8_0 Q4_K_M; do
  NEW=$G/.new-$N-$q.gguf
  if [ -f "$NEW" ] && [ ! -f "$NEW.complete" ]; then
    echo "  discarding incomplete $q ($(stat -c %s "$NEW") B)"
    rm -f "$NEW"
  fi
  if [ ! -f "$NEW" ]; then
    echo "=== $q ==="
    "$QZ" --token-embedding-type bf16 --output-tensor-type bf16 \
      --tensor-type-file "$G/tensor-types.txt" "$TMP" "$NEW" $q 8 2>&1 \
      | grep -E "blk\.(0|10|11)\.|token_embd|output\.weight|model size|quant size|error|fail" | head -40
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
      echo "  $q QUANTIZE FAILED rc=$rc"; rm -f "$NEW"
      echo "=== $(date -Is) REBUILD_FAILED ==="; exit 1
    fi
    touch "$NEW.complete"
  fi
done

echo "=== load test ==="
PASS=1
for q in Q8_0 Q4_K_M; do
  NEW=$G/.new-$N-$q.gguf
  [ -f "$NEW" ] || { echo "  $q MISSING"; PASS=0; continue; }
  echo "--- $q ---"
  timeout 1500 "$CLI" -m "$NEW" --no-warmup -t 6 -c 256 -n 24 --temp 0 --seed 1 \
    -st --simple-io -p "The capital of France is" 2>&1 | tail -14
  rc=${PIPESTATUS[0]}
  echo "  $q exit=$rc"
  [ "$rc" -eq 0 ] || PASS=0
done

if [ "$PASS" -ne 1 ]; then
  echo "=== LOAD TEST FAILED: leaving old files in place, nothing replaced ==="
  echo "=== $(date -Is) REBUILD_FAILED ==="
  exit 1
fi

echo "=== promoting ==="
for q in Q8_0 Q4_K_M; do
  mv -f "$G/.new-$N-$q.gguf" "$G/$N-$q.gguf"
  rm -f "$G/.new-$N-$q.gguf.complete"
  echo "  $q promoted"
done
rm -f "$G/.tmp-bf16.gguf"          # the old MTP-carrying intermediate, build scratch
rm -f "$TMP" "$TMP.complete"       # the no-MTP intermediate, also scratch
( cd "$G" && find . -maxdepth 1 -type f ! -name SHA256SUMS ! -name '.*' -printf '%P\n' \
    | sort | xargs -r sha256sum > SHA256SUMS )
echo "  gguf: $(wc -l < "$G/SHA256SUMS") files hashed"
ls -la "$G"
df -h "$Q" | tail -1
echo "=== $(date -Is) REBUILD_DONE ==="
