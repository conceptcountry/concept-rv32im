#!/usr/bin/env bash
# Build the distributable formats of Concept RV32IM 27B from the vision+MTP bf16 release
# layout. Mirrors ~/wsp/dist_build.sh and gguf_build.sh, which built Qwen Concept 27B.
# In every format the 11 processor layers, the embedding and the output head stay bf16.
set -u
Q=$HOME/qwenputer
OUT=$Q/dist
PY=${PY:-python}
QZ=$HOME/llama.cpp/build/bin/llama-quantize
CONV=$HOME/llama.cpp/convert_hf_to_gguf.py
N=concept-rv32im-27b
echo "=== $(date -Is) start ==="; df -h "$Q" | tail -1

for fmt in fp8 nvfp4a16; do
  if [ ! -f "$OUT/$fmt/config.json" ]; then
    echo "=== $fmt ==="
    "$PY" "$Q/rv_quant.py" $fmt "$OUT/bf16" "$OUT/$fmt" 2>&1 | tail -15 || echo "  $fmt FAILED"
    du -sh "$OUT/$fmt" 2>/dev/null
  fi
done

mkdir -p "$OUT/gguf"
TMP=$OUT/gguf/.tmp-bf16.gguf
if [ ! -f "$TMP" ] && [ ! -f "$OUT/gguf/$N-Q4_K_M.gguf" ]; then
  echo "=== bf16 gguf (intermediate) ==="
  "$PY" "$CONV" "$OUT/bf16" --outtype bf16 --outfile "$TMP" 2>&1 | tail -25
fi
if [ -f "$TMP" ]; then
  cat > "$OUT/gguf/tensor-types.txt" <<EOF
^blk\.([0-9]|10)\..*=bf16
EOF
  for q in Q8_0 Q4_K_M; do
    [ -f "$OUT/gguf/$N-$q.gguf" ] && continue
    echo "=== $q ==="
    "$QZ" --token-embedding-type bf16 --output-tensor-type bf16 \
      --tensor-type-file "$OUT/gguf/tensor-types.txt" "$TMP" "$OUT/gguf/$N-$q.gguf" $q 8 2>&1 \
      | grep -E "blk\.(0|10|11)\.|token_embd|output\.weight|model size|quant size|error|fail" | head -80
    ls -la "$OUT/gguf/$N-$q.gguf" 2>/dev/null
  done
  if [ ! -f "$OUT/gguf/mmproj-$N-bf16.gguf" ]; then
    echo "=== mmproj ==="
    "$PY" "$CONV" "$OUT/bf16" --mmproj --outtype bf16 --outfile "$OUT/gguf/mmproj-$N-bf16.gguf" 2>&1 | tail -8
  fi
else
  echo "  GGUF CONVERT FAILED"
fi

for d in bf16 fp8 nvfp4a16 gguf; do
  [ -d "$OUT/$d" ] || continue
  ( cd "$OUT/$d" && find . -maxdepth 1 -type f ! -name SHA256SUMS ! -name '.tmp*' -printf '%P\n' | sort | xargs -r sha256sum > SHA256SUMS )
  echo "  $d: $(wc -l < "$OUT/$d/SHA256SUMS") files hashed"
done
du -sh "$OUT"/* ; ls -la "$OUT/gguf"
echo "=== $(date -Is) BUILD_DONE ==="
