#!/usr/bin/env bash
# Build the MNIST inference program for the chipformer RV32 machine and for qemu.
# Mirrors the C-program flags in chipformer/rv/env/build_tests.sh.
#
#   build.sh [ENV_DIR] [MLP_DIR]
#
# Produces MLP_DIR/c-mlp.elf       (ROM/DROM/RAM map, for the reference and the circuit)
#          MLP_DIR/c-mlp.qemu.elf  (qemu-mappable, putchar as write(2))
set -euo pipefail

E=${1:-$HOME/qwenputer/qwenputer/experimental/riscv/chipformer/rv/env}
M=${2:-$HOME/qwenputer/mlp}

PICO=/usr/lib/picolibc/riscv64-unknown-elf
GCC=riscv64-unknown-elf-gcc
MARCH=rv32im_zba
LIBGCC=$($GCC -march=rv32im -mabi=ilp32 -print-libgcc-file-name)
LIBS="-Wl,--start-group $PICO/lib/release/rv32im/ilp32/libc.a $LIBGCC -Wl,--end-group"

for map in rom qemu; do
  sfx=$([ "$map" = rom ] && echo "" || echo ".qemu")
  def=$([ "$map" = rom ] && echo "" || echo "-DQEMU")
  $GCC -march=$MARCH -mabi=ilp32 -O2 $def -isystem "$PICO/include" -I"$E" -I"$M" \
    -nostdlib -nostartfiles -static -T"$E/$map.ld" -o "$M/c-mlp$sfx.elf" \
    "$E/crt0.S" "$M/mlp.c" "$E/sys.c" $LIBS
done

riscv64-unknown-elf-size -A "$M/c-mlp.elf" | head -6
echo "qemu reference run:"
qemu-riscv32 -cpu rv32,zba=true "$M/c-mlp.qemu.elf"
