#!/bin/bash


echo "running tests"


mkdir -p results
mkdir -p results/torch
mkdir -p results/tensorflow

# ------------------------------------------------------------- 

mkdir -p results/torch/effnet-v2-gpu
echo "running effnetv2 gpu"

sleep 10
sudo tegrastats --start --logfile effnetv2-torch-gpu.log &

sleep 120
docker run --name effnetv2gpu -e USE_GPU="true" -v ./results/torch/effnet-v2-gpu:/results:rw -v /home/student/research/OW-workloads/test-images/images-json-format:/images:ro effnetv2-torch:latest
docker container stop effnetv2gpu
docker container rm -f effnetv2gpu
sleep 120

sudo tegrastats --stop
sleep 10

echo "effnetv2 gpu done" 


# ------------------------------------------------------------- 

mkdir -p results/torch/resnet-50-gpu
echo "running resnet50 gpu"

sleep 10
sudo tegrastats --start --logfile resnet50-torch-gpu.log & 

sleep 120
docker run --name resnet50gpu -e USE_GPU="true" -v ./results/torch/resnet-50-gpu:/results:rw -v /home/student/research/OW-workloads/test-images/images-json-format:/images:ro resnet50-torch:latest
docker container stop resnet50gpu
docker container rm -f resnet50gpu
sleep 120

sudo tegrastats --stop
sleep 10

echo "resnet50 gpu done" 


# ------------------------------------------------------------- 

mkdir -p results/torch/effnet-v2-cpu
echo "running effnetv2 cpu"

sleep 10
sudo tegrastats --start --logfile effnetv2-torch-cpu.log &

sleep 120
docker run --name effnetv2cpu -e USE_GPU="false" -v ./results/torch/effnet-v2-cpu:/results:rw -v /home/student/research/OW-workloads/test-images/images-json-format:/images:ro effnetv2-torch:latest
docker container stop effnetv2cpu
docker container rm -f effnetv2cpu
sleep 120

sudo tegrastats --stop
sleep 10

echo "effnetv2 cpu done" 


# ------------------------------------------------------------- 

mkdir -p results/torch/resnet-50-cpu
echo "running resnet50 cpu"

sleep 10
sudo tegrastats --start --logfile resnet50-torch-cpu.log & 

sleep 120
docker run --name resnet50cpu -e USE_GPU="false" -v ./results/torch/resnet-50-cpu:/results:rw -v /home/student/research/OW-workloads/test-images/images-json-format:/images:ro resnet50-torch:latest
docker container stop resnet50cpu
docker container rm -f resnet50cpu
sleep 120

sudo tegrastats --stop
sleep 10

echo "resnet50 cpu done" 


# ------------------------------------------------------------- 

