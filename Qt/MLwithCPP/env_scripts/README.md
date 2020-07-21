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

3. Indentify active container ID
```
docker container ls
```

### Building Development Environment on Local Machine