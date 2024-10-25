#!/bin/bash

folders=('arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0086_PARISfinetunedSPLASHFPweight_PARIStest0100' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0086_PARISfinetunedSPLASHFPweight_PARIStest0101')

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

    nohup python run_binary_cl_on_test_500.py --folder="$folder" --how=val &> run_binary_cl_on_test_val.out &
    pid6=$!

    wait "$pid1"
    wait "$pid2"
    wait "$pid3"
    wait "$pid4"
    wait "$pid5"
    wait "$pid6"
done

# nohup ./run_multiple_folders_on_test500.sh &> run_multiple_folders_on_test500.out &




# nohup python run_binary_cl_on_test_500.py --folder="$folder" --how=train_val_fine_tuning &> run_binary_cl_on_test_500train_val_fine_tuning.out &
# pid7=$!

# wait "$pid7"
