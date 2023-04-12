#!/bin/bash

# Script used to synchronise files between a local machine and a remote server over SSH.
# The script supports two commands:
#  - push-changes: Syncs the local directory with the remote directory, excluding certain directories.
#  - fetch-runs: Syncs the remote directory with the local directory.
# The script uses rsync to transfer files and directories efficiently and securely.

if [ $# -ne 3 ]; then
  echo "Usage: $0 <ssh_host> <ssh_port> <command>"
  echo "Commands: push-changes, fetch-runs"
  exit 1
fi

ssh_host=$1
ssh_port=$2
command=$3

ssh_user="root"


local_dir=".."
remote_dir="/root/zindi_tree_counting_competition/"



run_command() {
    command=$1
    echo -n "${command}... "
    start_time=$(date +%s)
    eval "${command}"
    end_time=$(date +%s)
    echo "Done. Time taken: $(($end_time-$start_time)) seconds."
}

if [ "$command" = "push-changes" ]; then

  exclude_dirs=("data" "runs/*/" "logs" "notebooks" "preds" ".git" ".idea")
  exclude_opts=$(printf -- "--exclude '%s' " "${exclude_dirs[@]}")

  echo -e "\n\n\n\n\nPushing latest changes... $exclude_opts\n"
  run_command "rsync -avz $exclude_opts --update --inplace -e 'ssh -p $ssh_port' $local_dir $ssh_user@$ssh_host:$remote_dir"

elif [ "$command" = "fetch-runs" ]; then

  exclude_dirs=("")
  exclude_opts=$(printf -- "--exclude '%s' " "${exclude_dirs[@]}")

  echo -e "\n\n\n\n\n1. Syncing latest changes... $exclude_opts\n"
  run_command "rsync -avz -e 'ssh -p $ssh_port' $ssh_user@$ssh_host:$remote_dir $local_dir"

else
  echo "Invalid command: $command"
  exit 1
fi