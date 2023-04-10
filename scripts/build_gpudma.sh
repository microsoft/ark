# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/usr/bin/env bash

perr() {
    >&2 echo "ERROR: $1"
}

ck() {
    if [ $? != 0 ]; then
        exit 1
    fi
}

ex() {
    echo "\$ ${@/eval/}"
    eval "$@"
    ck
}

if [ -z ${ARKDIR+x} ]; then
    perr "ARKDIR is unset."
    exit 1
fi

TMPDIR=$(mktemp -d)
NVIDIA_SRC=$(ls -d1 /usr/src/nvidia-*.*.* | tail -n 1)

ex "mkdir -p $TMPDIR/nvidia"
ex "cp -r $NVIDIA_SRC $TMPDIR/nvidia/kernel"
ex "make -j -C $TMPDIR/nvidia/kernel"
ex "cd $ARKDIR/third_party/gpudma/module && GPUDMA_DIR=$TMPDIR make"
ex "rm -r $TMPDIR"
