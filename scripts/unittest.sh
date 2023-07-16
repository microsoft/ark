#!/usr/bin/env bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

perr() {
    >&2 echo "ERROR: $1"
}

ex() {
    echo "\$ ${@/eval/}"
    eval "$@"
}

ck() {
    ret=$?
    if [ $ret != 0 ]; then
        perr "$ret"
        exit 1
    fi
}

help()
{
   echo ""
   echo "Usage: $0 [-g] [-t TEST_NAME]"
   echo -e "\t-g (Optional) Run with gdb."
   echo -e "\t-m (Optional) Run with cuda-memcheck."
   echo -e "\t-t (Optional) Name of the test to run."
   exit 1
}

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname ${SCRIPT})
ARK_BUILD_DIR=${SCRIPT_DIR}/../build/ark
DEFAULT_ARK_ROOT="$(realpath ${SCRIPT_DIR}/../build)"

if [ -z ${ARK_ROOT+x} ]; then
    echo "ARK_ROOT is unset - select default path $DEFAULT_ARK_ROOT"
    ARK_ROOT=$DEFAULT_ARK_ROOT
fi

# Get command line arguments
while getopts "mgt:" opt; do
    case "$opt" in
        g ) GDB="gdb -q" ;;
        m ) CM="/usr/local/cuda/bin/cuda-memcheck" ;;
        t ) TEST_NAME="$OPTARG" ;;
        ? ) help ;;
    esac
done

ex "rm /dev/shm/ark.* 2> /dev/null"
ex "sysctl fs.inotify.max_user_watches=524288"

ex "nvidia-smi -pm 1"
for i in $(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 ))); do
    ex "nvidia-smi -ac $(nvidia-smi --query-gpu=clocks.max.memory,clocks.max.sm --format=csv,noheader,nounits -i $i | sed 's/\ //') -i $i"
done

PREFIX="ARK_ROOT=$ARK_ROOT LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ARK_ROOT/lib $CM $GDB"

if [ -z $TEST_NAME ]; then
    ex "$PREFIX $ARK_BUILD_DIR/ipc/ipc_mem_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/ipc/ipc_coll_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/ipc/ipc_socket_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/gpu/gpu_mem_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/gpu/gpu_mgr_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/net/net_ib_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/gpu/gpu_kernel_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/ops/ops_tensor_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/ops/ops_identity_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/ops/ops_reshape_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/ops/ops_dot_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/ops/ops_add_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/ops/ops_mul_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/ops/ops_scale_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/ops/ops_gelu_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/ops/ops_reduce_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/ops/ops_matmul_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/ops/ops_im2col_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/ops/ops_transpose_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/ops/ops_all_reduce_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/dims_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/sched/sched_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/ops/ops_sendrecv_test"; ck;
    ex "$PREFIX $ARK_BUILD_DIR/ops/ops_sendrecv_mm_test"; ck;
else
    ex "$PREFIX $ARK_BUILD_DIR/$TEST_NAME"; ck;
fi
