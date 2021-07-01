#!/bin/bash
files=$(ssh cluster "ls -1 ~/code/second.pytorch/second/ | grep 'model_dir_car'")
for f in $files
do
    summary_folder="cluster_summaries/$f/"
    echo "$summary_folder"
    mkdir -p "$summary_folder"
    scp cluster:"~/code/second.pytorch/second/$f/summary/*" "$summary_folder"
done