#!/usr/bin/env bash

set -e

declare -A buildImageTable
buildImageTable=(
    ["cuda11.8"]="nvidia/cuda:11.8.0-devel-ubuntu20.04"
    ["cuda12.1"]="nvidia/cuda:12.1.1-devel-ubuntu20.04"
    ["cuda12.2"]="nvidia/cuda:12.2.2-devel-ubuntu20.04"
    ["cuda12.4"]="nvidia/cuda:12.4.1-devel-ubuntu20.04"
    ["rocm5.7"]="rocm/dev-ubuntu-20.04:5.7"
    ["rocm6.0"]="rocm/dev-ubuntu-20.04:6.0"
    ["rocm6.1"]="rocm/dev-ubuntu-20.04:6.1"
)

declare -A baseImageTable
baseImageTable=(
    ["cuda11.8"]="nvidia/cuda:11.8.0-devel-ubuntu20.04"
    ["cuda12.1"]="nvidia/cuda:12.1.1-devel-ubuntu20.04"
    ["cuda12.2"]="nvidia/cuda:12.2.2-devel-ubuntu20.04"
    ["cuda12.4"]="nvidia/cuda:12.4.1-devel-ubuntu20.04"
    ["rocm5.7"]="rocm/dev-ubuntu-20.04:5.7-complete"
    ["rocm6.0"]="rocm/dev-ubuntu-20.04:6.0-complete"
    ["rocm6.1"]="rocm/dev-ubuntu-20.04:6.1-complete"
)

declare -A extraLdPathTable
extraLdPathTable=(
    ["cuda11.8"]="/usr/local/cuda-11.8/lib64"
    ["cuda12.1"]="/usr/local/cuda-12.1/compat:/usr/local/cuda-12.1/lib64"
    ["cuda12.2"]="/usr/local/cuda-12.2/compat:/usr/local/cuda-12.2/lib64"
    ["cuda12.4"]="/usr/local/cuda-12.4/compat:/usr/local/cuda-12.4/lib64"
    ["rocm5.7"]="/opt/rocm/lib"
    ["rocm6.0"]="/opt/rocm/lib"
    ["rocm6.1"]="/opt/rocm/lib"
)

GHCR="ghcr.io/microsoft/ark/ark"
TARGET=${1}

print_usage() {
    echo "Usage: $0 [cuda11.8|cuda12.1|cuda12.2|cuda12.4|rocm5.7|rocm6.0|rocm6.1]"
}

if [[ ! -v "baseImageTable[${TARGET}]" ]]; then
    echo "Invalid target: ${TARGET}"
    print_usage
    exit 1
fi
echo "Target: ${TARGET}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd ${SCRIPT_DIR}/..

docker build -t ${GHCR}:build-${TARGET} \
    -f docker/build-x.dockerfile \
    --build-arg BASE_IMAGE=${buildImageTable[${TARGET}]} \
    --build-arg EXTRA_LD_PATH=${extraLdPathTable[${TARGET}]} .

docker build -t ${GHCR}:base-${TARGET} \
    -f docker/base-x.dockerfile \
    --build-arg BASE_IMAGE=${baseImageTable[${TARGET}]} \
    --build-arg EXTRA_LD_PATH=${extraLdPathTable[${TARGET}]} .

docker build -t ${GHCR}:base-dev-${TARGET} \
    -f docker/base-dev-x.dockerfile \
    --build-arg BASE_IMAGE=${GHCR}:base-${TARGET} .
