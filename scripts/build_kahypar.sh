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

# Create build dir
if [ -z ${ARKDIR+x} ]; then
    perr "ARKDIR is unset."
    exit 1
fi
if [ -z ${BDIR+x} ]; then
    perr "BDIR is unset."
    exit 1
fi
WRK=$BDIR/third_party/kahypar
test -e $WRK
if [ $? != 0 ]; then
    ex "mkdir -p $WRK"
fi

# Test whether libboost is already built.
test -e $WRK/boost_1_69_0/stage/lib/libboost_program_options.so.1.69.0
if [ $? != 0 ]; then
    # Download and build Boost 1.69 (for KaHyPar)
    test -e $WRK/boost_1_69_0/
    if [ $? != 0 ]; then
        DL_PATH="$WRK/boost_1_69_0.tar.gz"
        ex "curl --create-dirs -L -C- https://jaist.dl.sourceforge.net/project/boost/boost/1.69.0/boost_1_69_0.tar.gz -o $DL_PATH"
        ex "tar xzf $DL_PATH -C $WRK"
        ex "rm $DL_PATH"
    fi
    ex "cd $WRK/boost_1_69_0/"
    ex "./bootstrap.sh --with-libraries=program_options"
    ex "./b2 -j 40 --with-program_options"
fi
BOOST_ROOT="$WRK/boost_1_69_0"

# Test whether libkahypar is already built.
test -e $WRK/kahypar/build/install/lib/libkahypar.so
if [ $? != 0 ]; then
    # Download and build cmake-3.16.5 (for building KaHyPar)
    test -e $WRK/cmake-3.16.5/
    if [ $? != 0 ]; then
        DL_PATH="$WRK/cmake-3.16.5.tar.gz"
        ex "curl --create-dirs -L -C- https://cmake.org/files/v3.16/cmake-3.16.5.tar.gz -o $DL_PATH"
        ex "tar xzf $DL_PATH -C $WRK"
        ex "rm $DL_PATH"
    fi
    test -e $WRK/cmake-3.16.5/bin/cmake
    if [ $? != 0 ]; then
        ex "cd $WRK/cmake-3.16.5/"
        ex "./bootstrap -- -DCMAKE_USE_OPENSSL=OFF"
        ex "make -j"
    fi
    CMAKE="$WRK/cmake-3.16.5/bin/cmake"

    # Download and build KaHyPar
    test -e $WRK/kahypar
    if [ $? != 0 ]; then
        ex "cd $ARKDIR && git submodule update --init --recursive"
        ex "cp -r $ARKDIR/third_party/kahypar $WRK"
    fi
    ex "cd $WRK/kahypar"
    ex "mkdir -p build && cd build"
    PYTHON_INC=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")
    ck
    PYTHON_LIBDIR=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
    ck
    ex "$CMAKE .. \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_CXX_FLAGS=-std=c++14 \
        -DCMAKE_INSTALL_PREFIX=$PWD/install \
        -DBOOST_ROOT=$BOOST_ROOT \
        -DPYTHON_INCLUDE_DIR=$PYTHON_INC \
        -DPYTHON_LIBRARY=$PYTHON_LIBDIR"
    ex "make install.library"
fi
