#!/bin/bash

folders=('arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0030' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0036' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0037' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0038' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0041' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0042' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0043' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0044' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0045' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0047' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0048' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0050' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0051' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0053' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0055' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0057' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0058' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0060' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0061' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0068' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0072' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0074' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0076' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0077' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0079' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0081' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0082' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedSPLASH_PARIStest0084')

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
