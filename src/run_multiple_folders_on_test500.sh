#!/bin/bash

folders=('arch2_PARIStrained_PARISval0071')

for folder in "${folders[@]}"; do

    nohup python run_binary_cl_on_test_500.py --folder="$folder" &> run_binary_cl_on_test_500.out &
    pid1=$!

    nohup python run_binary_cl_on_test_500.py --folder="$folder" --hq &> run_binary_cl_on_testHQ.out &
    pid2=$!

    nohup python run_binary_cl_on_test_500.py --folder="$folder" --dataset=ricseq &> run_binary_cl_on_test_ricseq500.out &
    pid3=$!
    
    nohup python run_binary_cl_on_test_500.py --folder="$folder" --dataset=mario &> run_binary_cl_on_test_mario500.out &
    pid4=$!

    nohup python run_binary_cl_on_test_500.py --folder="$folder" --dataset=splash &> run_binary_cl_on_test_splash500.out &
    pid5=$!

    wait "$pid1"
    wait "$pid2"
    wait "$pid3"
    wait "$pid4"
    wait "$pid5"
done

# nohup ./run_multiple_folders_on_test500.sh &> run_multiple_folders_on_test500.out &



#  # Wait for each command within the folder iteration to complete
# wait "$pid1"
