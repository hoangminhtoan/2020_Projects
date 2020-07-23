#!/usr/bin/env bash
set -x
set -e

DEV_DIR=$(cd ../development && pwd)
CUR_DIR=$(pwd)
REPOSITORY=$1
COMMIT_HASH=$2

cd $DEV_DIR
git clone $REPOSITORY
cd "$(basename "$REPOSITORY" .git)"
git checkout $COMMIT_HASH
git submodule update --init --recursive
cd $CUR_DIR