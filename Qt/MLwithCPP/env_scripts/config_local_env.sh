apt-get install -y build-essential
apt-get install -y gdb
apt-get install -y git
apt-get install -y cmake
apt-get install -y cmake-curses-gui
apt-get install -y python
apt-get install -y python-pip
apt-get install -y libblas-dev
apt-get install -y libopenblas-dev
apt-get install -y libatlas-base-dev
apt-get install -y liblapack-dev
apt-get install -y libboost-all-dev
apt-get install -y libopencv-core3.2
apt-get install -y libopencv-imgproc3.2
apt-get install -q -y libopencv-dev
apt-get install -y libopencv-highgui3.2
apt-get install -y libopencv-highgui-dev
apt-get install -y libhdf5-dev
apt-get install -y libjson-c-dev
apt-get install -y libx11-dev
apt-get install -y openjdk-8-jdk
apt-get install -y wget
apt-get install -y ninja-build
apt-get install -y gnuplot
apt-get install -y vim
apt-get install -y python3-venv
pip install pyyaml
RUN pip install typing

cd ~/
mkdir development
cd ~/development
cp /path/to/examples/package/docker/checkout_lib.sh ~/development
cp /path/to/examples/package/docker/install_lib.sh ~/development
cp /path/to/examples/package/docker/install_env.sh ~/development
cp /path/to/examples/package/docker/install_android.sh ~/development
chmod 777 ~/development/checkout_lib.sh
chmod 777 ~/development/install_lib.sh
chmod 777 ~/development/install_env.sh
chmod 777 ~/development/install_android.sh
./install_env.sh

