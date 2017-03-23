#!/usr/bin/env sh
set -e

TOOLS=./build_py/tools

$TOOLS/caffe train --solver=examples/tes/train_siamese_solver.prototxt $@
