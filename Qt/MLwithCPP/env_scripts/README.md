# Building development environment

## Table of Contents
 * [Building Development Environment with Docker](#building-development-environment-with-docker)
 * [Building Development Environment on Local Machine](#building-development-environment-on-local-machine)



### Building Development Environment with Docker
1. Build Docker Image.
```
cd env_scripts
docker build -t buildenv:1.0
```

2. Run the Docker container and build third party libs there.
```
docker run -it buildenv:1.0 bash
cd /development
./install_env.sh
./install_android.sh
```

3. Indentify active container ID & Save the container as a new image
```
docker container ls
docker commit [container ID]
```

4. Indentify new image ID
```
docker image ls
```

5. Give the name of the new image
```
docker tag [image ID] [new name]
```

6. Stop the initial container in the original console session
```
exit
```

7. Run a new container from the created image, and share code samples folder
```
docker run -it -v [host_path]:[container_path] [new name] bash
```

8. Samples from chapter 2 require accsess to your graphical environment to show images. You can share you X11 server with a Docker container. The following script shows how to run a container with graphics environment:

```
xhost +local: root
docker run --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp.X11-unix -it -v [host_path]:[container_path] [new name] bash
```

9. In the stated container console session
```
./build_ch1.sh
```

### Building Development Environment on Local Machine
1. Install Ubuntu 18.04

2. Configuring Local Environment
```
bash config_local_env.sh
```

3. All third-party libs will be installed into the following directory
$HOME/development/libs

4. Navigate to the ```/path/to/examples/package/build_scripts``` folder

5. Select the build scripts for the chapter 

6. Updated the ```LIBS_DIR``` variable in the script with the ```$HOME/development/libs``` value, or another one but it should the folder where all third-party libs are installed

7. Run the build scrip to compile samples for the selected chapter

#### List of all third-party libraries
Name | Commit Hash | Branch name | Repository
Shotgun | f7255cf2cc6b5116e50840816d70d21e7cc039bb | master | https://github.com/shogun-toolbox/shogun
SharkML | 221c1f2e8abfffadbf3c5ef7cf324bc6dc9b4315 | master - https://github.com/Shark-ML/Shark

Armadillo | 442d52ba052115b32035a6e7dc6587bb6a462dec| branch 9.500.x | https://gitlab.com/conradsnicta/armadillo-code

DLib | 929c630b381d444bbf5d7aa622e3decc7785ddb2 | v19.15 | https://github.com/davisking/dlib

Eigen | cf794d3b741a6278df169e58461f8529f43bce5d | 3.3.7 | https://github.com/eigenteam/eigen-git-mirror

mlpack - e2f696cfd5b7ccda2d3af1c7c728483ea6591718 - master - https://github.com/mlpack/mlpack

plotcpp - c86bd4f5d9029986f0d5f368450d79f0dd32c7e4 - master - https://github.com/Kolkir/plotcpp

PyTorch - 8554416a199c4cec01c60c7015d8301d2bb39b64 - v1.2.0 - https://github.com/pytorch/pytorch

xtensor - 02d8039a58828db1ffdd2c60fb9b378131c295a2 - master - https://github.com/xtensor-stack/xtensor

xtensor-blas - 89d9df93ff7306c32997e8bb8b1ff02534d7df2e - master - https://github.com/xtensor-stack/xtensor-blas

xtl - 03a6827c9e402736506f3ded754e890b3ea28a98 - master - https://github.com/xtensor-stack/xtl

OpenCV 3 - from the distribution installation package - https://github.com/opencv/opencv_contrib/releases/tag/3.3.0

fast-cpp-csv-parser - 3b439a664090681931c6ace78dcedac6d3a3907e - master - https://github.com/ben-strasser/fast-cpp-csv-parser

RapidJson | 73063f5002612c6bf64fe24f851cd5cc0d83eef9 | master | https://github.com/Tencent/rapidjson
