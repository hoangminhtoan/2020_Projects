# Run the docker with all GPU(s) 
docker run -d \
-it \
--name dl_with_cpp \
--gpus all \
-v /media/toanmh/Workspace/Github/2020_Projects/Qt/MLwithCPP:/workspace/ml_with_cpp \
ml_cpp:1.0

# Run the docker with specific GPU(s)
#docker run -d \
#-it \
#--name {your_container_name} \
#--gpus devices=0 \
#-v {host_path}:/{docker_path} \
#-t {image}