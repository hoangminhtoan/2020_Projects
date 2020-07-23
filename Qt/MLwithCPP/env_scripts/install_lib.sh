#!/usr/bin/env bash
set -x
set -e

DEV_DIR=$(cd ../development && pwd)
CUR_DIR=$(pwd)
REPOSITORY=$1
COMMIT_HASH=$2
shift; shift;
EXTRA_CMAKE_PARAMS=$@

cd $DEV_DIR
git clone $REPOSITORY
cd "$(basename "$REPOSITORY" .git)"
git checkout $COMMIT_HASH
git submodule update --init --recursive
mkdir build
cd build 
cmake -DCMAKE_INSTALL_PREFIX=$DEV_DIR/libs $EXTRA_CMAKE_PARAMS ..
cmake --build . --target install -- -j8
cd ..
rm -rf build
cd $CUR_DIR