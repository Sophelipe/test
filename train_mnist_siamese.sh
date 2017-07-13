#!/usr/bin/env sh
set -e

TOOLS=./build_py/tools
GLOG_logtostderr=0 GLOG_log_dir=examples/tes/log/train2/ \
$TOOLS/caffe train --solver=examples/tes/train_siamese_solver.prototxt \
--snapshot=examples/tes/train1/train2_iter_160836.solverstate

