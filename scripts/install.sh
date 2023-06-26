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
        perr "$1 ($ret)"
        exit 1
    fi
}

if [ -z ${ARKDIR+x} ]; then
    perr "ARKDIR is unset."
    exit 1
fi

# Select install path.
DEFAULT_ARK_ROOT=/usr/local/ark
if [ -z ${ARK_ROOT+x} ] || [ "$ARK_ROOT" == "" ]; then
    echo "ARK_ROOT is unset -- select default path $DEFAULT_ARK_ROOT"
    ARK_ROOT=$DEFAULT_ARK_ROOT
else
    echo "ARK_ROOT=$ARK_ROOT"
fi
test -e $ARK_ROOT
if [ $? != 0 ]; then
    mkdir -p $ARK_ROOT
    ck "Failed to create $ARK_ROOT"
fi

KAHYPAR_BUILD_PATH="$ARKDIR/build/third_party/kahypar"
KAHYPAR_INI_PATH="$ARKDIR/third_party/kahypar/config/cut_kKaHyPar_dissertation.ini"

# Include files.
INCLUDE="$BDIR/include"

# Library files.
LIB="$BDIR/lib"
USE_KAHYPAR=${USE_KAHYPAR:-false}
if $USE_KAHYPAR; then
    KAHYPAR_LIB="${KAHYPAR_BUILD_PATH}/kahypar/build/install/lib/libkahypar.so \
    $KAHYPAR_BUILD_PATH/boost_1_69_0/stage/lib/libboost_program_options.so.1.69.0"
else
    KAHYPAR_LIB=""
fi

# Test whether all files exist in build.
for path in $INCLUDE $LIB; do
    test -e $path
    ck "Build is incomplete."
done

# Setup the install directory.
mkdir -p $ARK_ROOT/include
mkdir -p $ARK_ROOT/lib

# Copy files into the install directory.
ex "rsync -ar --delete $INCLUDE/* $ARK_ROOT/include"
ck "Failed to copy include files."
ex "rsync -ar --delete $LIB/* $ARK_ROOT/lib"
ck "Failed to copy library files."
if $USE_KAHYPAR; then
    ex "rsync -a $KAHYPAR_INI_PATH $ARK_ROOT"
    ck "Failed to copy kahypar ini file."
    ex "rsync -a $KAHYPAR_LIB $ARK_ROOT/lib"
    ck "Failed to copy kahypar library files."
fi

test -e $ARK_ROOT/hostfile
if [ $? != 0 ]; then
    # Create a default hostfile.
    echo "127.0.0.1" > $ARK_ROOT/hostfile
    ck "Failed to create hostfile."
fi

echo "ARK installation succeed: ARK_ROOT=$ARK_ROOT"
echo "Please add $ARK_ROOT/lib to LD_LIBRARY_PATH"
