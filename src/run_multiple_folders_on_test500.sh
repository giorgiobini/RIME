#!/bin/bash

folders=('all_modelarch2_easypretrain11_paristfinetuningSPLASHval0039parisHQSPLASHval0043' 'all_modelarch2_easypretrain11_paristfinetuningSPLASHval0039parisHQSPLASHval0054' 'all_modelarch2_easypretrain11_paristfinetuningSPLASHval0039parisHQSPLASHval0080' 'all_modelarch2_easypretrain11_paristfinetuningSPLASHval0039parisHQSPLASHval0100' 'all_modelarch2_easypretrain11_paristfinetuningSPLASHval0039parisHQSPLASHval0103' 'all_modelarch2_easypretrain11_paristfinetuningSPLASHval0039parisHQSPLASHval0120' 'all_modelarch2_easypretrain11_paristfinetuningSPLASHval0061parisHQSPLASHval0070' 'all_modelarch2_easypretrain11_paristfinetuningSPLASHval0061parisHQSPLASHval0075' 'all_modelarch2_easypretrain11_paristfinetuningSPLASHval0061parisHQSPLASHval0085' 'all_modelarch2_easypretrain11_paristfinetuningSPLASHval0061parisHQSPLASHval0088' 'all_modelarch2_easypretrain11_paristfinetuningSPLASHval0061parisHQSPLASHval0098' 'all_modelarch2_easypretrain11_paristfinetuningSPLASHval0061parisHQSPLASHval0142' 'all_modelarch1_easypretrain10_paristfinetuningsplashRICSEQval0019' 'all_modelarch1_easypretrain10_paristfinetuningsplashRICSEQval0029' 'all_modelarch1_easypretrain10_paristfinetuningsplashRICSEQval0031' 'all_modelarch1_easypretrain10_paristfinetuningsplashRICSEQval0037' 'all_modelarch1_easypretrain10_paristfinetuningsplashRICSEQval0050' 'all_modelarch1_easypretrain10_paristfinetuningsplashRICSEQval0057' 'all_modelarch1_easypretrain10_paristfinetuningsplashRICSEQval0070')

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
