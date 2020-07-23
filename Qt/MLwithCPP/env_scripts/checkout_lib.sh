#!/usr/bin/env bash
set -x
set -e

START_DIR=$(cd ../development && pwd)
REPOSITORY=$1
COMMIT_HASH=$2

cd $START_DIR
git clone $REPOSITORY
cd "$(basename "$REPOSITORY" .git)"
git checkout $COMMIT_HASH
git submodule update --init --recursive
cd $START_DIR