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


cd $START_DIR/Chapter01/sharkml_samples/
mkdir build
cd build/
cmake -DSHARK_PATH=$LIBS_DIR ..
cmake --build . --target all


cd $START_DIR/Chapter01/xtensor_samples/
mkdir build
cd build/
cmake -DSHARK_PATH=$LIBS_DIR ..
cmake --build . --target all