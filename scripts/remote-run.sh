#!/bin/bash

# Script for running multiple instances of training script with different parameter files.

remote_dir="/root/zindi_tree_counting_competition"

net="efficientnet"

command="train"

# list all params files to run
params_list=("params.yaml")

for params_file in "${params_list[@]}"
do
    echo -e  "\n\n\n\n\n\nRunning with $params_file."
    start_time=$(date +%s)
    cd $remote_dir && /opt/conda/bin/python -m main --params_file $params_file $command --net $net
    end_time=$(date +%s)
    echo -e "\n\n\n\n\n\nFinished training with $params_file. Time taken: $(($end_time-$start_time)) seconds."
done
