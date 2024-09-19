#!/bin/bash

folders=('arch2_PARIStrained_PARISval0016' 'arch2_PARIStrained_PARISval0024' 'arch2_PARIStrained_PARISval0027' 'arch2_PARIStrained_PARISval0037' 'arch2_PARIStrained_PARISval0039' 'arch2_PARIStrained_PARISval0045' 'arch2_PARIStrained_PARISval0047' 'arch2_PARIStrained_PARISval0050' 'arch2_PARIStrained_PARISval0053' 'arch2_PARIStrained_PARISval0056' 'arch2_PARIStrained_PARISval0057' 'arch2_PARIStrained_PARISval0058' 'arch2_PARIStrained_PARISval0070' 'arch2_PARIStrained_PARISval0075' 'arch2_PARIStrained_PARISval0077' 'arch2_PARIStrained_PARISval0078' 'arch2_PARIStrained_PARISval0085' 'arch2_PARIStrained_PARISval0094')

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
