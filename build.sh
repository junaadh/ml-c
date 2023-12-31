#!/bin/sh

set -xe
doc="--doc"

if [ "$1" = "$doc" ]; then
  mkdir -p doc/
  # mkdir -p build/temp

  # pdflatex -halt-on-error -output-directory=doc src/doc/grad.tex
  # mv doc/*.log doc/*.aux build/temp
  typst compile src/doc/grad.typ doc/grad2.pdf
else
  mkdir -p build/prev
  
  clang -Wall -Wextra -o build/prev/double src/prev/double.c -lm
  clang -Wall -Wextra -o build/prev/gates src/prev/gates.c -lm
  clang -Wall -Wextra -o build/prev/xor src/prev/xor.c -lm
  clang -g -Wall -Wextra -o build/gates src/gates.c -lm
fi
