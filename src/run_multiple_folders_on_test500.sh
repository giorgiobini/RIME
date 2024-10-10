#!/bin/bash

folders=('arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0025' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0030' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0035' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0036' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0038' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0042' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0043' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0045' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0046' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0047' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0050' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0051' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0052' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0054' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0055' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0056' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0057' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0062' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0063' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0066' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0067' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0068' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0069' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0072' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0074' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0079' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0080' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0081' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0082' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0086' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0087' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0089' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0092' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0098' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0104' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0105')

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
