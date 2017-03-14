#!/usr/bin/env sh
set -e

TOOLS=D:/LHF/caffe/caffe/Build/x64/Release

$TOOLS/caffe train --solver=train_siamese_solver.prototxt $@
