#!/bin/bash

folders=('all_modelarch2_easypretrain11_paristfinetuningsplashRICSEQval0013' 'all_modelarch2_easypretrain11_paristfinetuningsplashRICSEQval0018' 'all_modelarch2_easypretrain11_paristfinetuningsplashRICSEQval0019' 'all_modelarch2_easypretrain11_paristfinetuningsplashRICSEQval0020' 'all_modelarch2_easypretrain11_paristfinetuningsplashRICSEQval0025' 'all_modelarch2_easypretrain11_paristfinetuningsplashRICSEQval0028' 'all_modelarch2_easypretrain11_paristfinetuningsplashRICSEQval0031' 'all_modelarch2_easypretrain11_paristfinetuningsplashRICSEQval0033' 'all_modelarch2_easypretrain11_paristfinetuningsplashRICSEQval0054' 'all_modelarch2_easypretrain11_paristfinetuningsplashRICSEQval0064' 'all_modelarch2_easypretrain11_paristfinetuningsplashRICSEQval0098')

for folder in "${folders[@]}"; do
    nohup python run_binary_cl_on_test_500.py --folder="$folder" &> run_binary_cl_on_test_500.out &
    pid1=$!

    nohup python run_binary_cl_on_test_500.py --folder="$folder" --dataset=mario &> run_binary_cl_on_test_mario500.out &
    pid2=$!

    nohup python run_binary_cl_on_test_500.py --folder="$folder" --dataset=ricseq &> run_binary_cl_on_test_ricseq500.out &
    pid3=$!

    nohup python run_binary_cl_on_test_500.py --folder="$folder" --dataset=splash &> run_binary_cl_on_test_splash500.out &
    pid4=$!

    # Wait for each command within the folder iteration to complete
    wait "$pid1"
    wait "$pid2"
    wait "$pid3"
    wait "$pid4"
done

# nohup ./run_multiple_folders_on_test500.sh &> run_multiple_folders_on_test500.out &
