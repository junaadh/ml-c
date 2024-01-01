#!/bin/sh

set -xe

mkdir -p build/prev

clang -Wall -Wextra -o build/prev/double src/prev/double.c -lm
clang -Wall -Wextra -o build/prev/gates src/prev/gates.c -lm
clang -Wall -Wextra -o build/prev/xor src/prev/xor.c -lm

clang -Wall -Wextra -o build/ars src/ars.c -lm
