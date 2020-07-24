# Parent directory
START_DIR=${PWD%/*}

# Libraries directory
LIBS_DIR=$START_DIR/development/libs

# Chapter 01
cd $START_DIR/Chapter01/dlib_samples/
mkdir build 
cd build/
cmake -DDLIB_PATH=$LIBS_DIR ..
cmake --build . --target all