#!/bin/bash

# Script for setting up a remote environment for the Zindi tree counting competition
# Requires the following arguments: "ssh_host", "ssh_port", "auth_token"

if [ $# -ne 3 ]; then
  echo "$0 requires arguments: \"ssh_host\", \"ssh_port\", \"auth_token\" "
  exit 1
fi

ssh_host=$1
ssh_port=$2
auth_token=$3

ssh_user="root"


local_dir=".."
remote_dir="/root/zindi_tree_counting_competition"

#  exclude runs directories but not `runs.csv`
exclude_dirs=("data" "runs/*/" "logs" "notebooks" "preds" ".git" ".idea")
exclude_opts=$(printf -- "--exclude '%s' " "${exclude_dirs[@]}")

run_command() {
    command=$1
    echo -n "${command}... "
    start_time=$(date +%s)
    eval "${command}"
    end_time=$(date +%s)
    echo "Done. Time taken: $(($end_time-$start_time)) seconds."
}

echo -e "\n\n\n\n\n1. Copying files to remote host... $exclude_opts\n"
run_command "rsync -avz $exclude_opts -e 'ssh -p $ssh_port' $local_dir $ssh_user@$ssh_host:$remote_dir"


echo -e  "\n\n\n\n\n2. Install opencv library if not present...\n"
run_command "ssh -p $ssh_port $ssh_user@$ssh_host \
             'apt-get update && apt-get install libgl1 -y'"

echo -e  "\n\n\n\n\n3. Installing requirements...\n"
run_command "ssh -p $ssh_port $ssh_user@$ssh_host \
             'cd $remote_dir && \
             /opt/conda/bin/pip install pandas opencv-python scikit-learn efficientnet_pytorch'"

echo -e "\n\n\n\n\n4. Downloading project data...\n"
run_command "ssh -p $ssh_port $ssh_user@$ssh_host \
             'cd $remote_dir && /opt/conda/bin/python -m main data_downloader --auth_token $auth_token'"

echo -e "\n\n\n\n\n5. Finished install."