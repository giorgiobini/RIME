#!/bin/bash

folders=('arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0036' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0042' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0045' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0047' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0051' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0054' 'arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0057' 'arch2_PARIStrained_PARISval0016' 'arch2_PARIStrained_PARISval0024' 'arch2_PARIStrained_PARISval0027' 'arch2_PARIStrained_PARISval0037' 'arch2_PARIStrained_PARISval0039' 'arch2_PARIStrained_PARISval0045' 'arch2_PARIStrained_PARISval0047' 'arch2_PARIStrained_PARISval0050' 'arch2_PARIStrained_PARISval0053' 'arch2_PARIStrained_PARISval0056' 'arch2_PARIStrained_PARISval0057' 'arch2_PARIStrained_PARISval0058' 'arch2_PARIStrained_PARISval0070' 'arch2_PARIStrained_PARISval0075' 'arch2_PARIStrained_PARISval0077' 'arch2_PARIStrained_PARISval0078' 'arch2_PARIStrained_PARISval0085' 'arch2_PARIStrained_PARISval0094' 'arch2_PARIStrained_PARISval0071' 'arch2_PARISfinetuned_PARIStest0013' 'arch2_PARISfinetuned_PARIStest0016' 'arch2_PARISfinetuned_PARIStest0023' 'arch2_PARISfinetuned_PARIStest0026' 'arch2_PARISfinetuned_PARIStest0033' 'arch2_PARISfinetuned_PARIStest0035' 'arch2_PARISfinetuned_PARIStest0043' 'arch2_PARISfinetuned_PARIStest0047' 'arch2_PARISfinetuned_PARIStest0050' 'arch2_PARISfinetuned_PARIStest0055' 'arch2_PARISfinetuned_PARIStest0057' 'arch2_PARISfinetuned_PARIStest0061' 'arch2_PARISfinetuned_PARIStest0068' 'arch2_PARISfinetuned_PARIStest0078')

for folder in "${folders[@]}"; do

    nohup python run_binary_cl_on_test_500.py --folder="$folder" --how=train_val_fine_tuning &> run_binary_cl_on_test_500train_val_fine_tuning.out &
    pid1=$!

    nohup python run_binary_cl_on_test_500.py --folder="$folder" --how=val &> run_binary_cl_on_test_val.out &
    pid2=$!

    wait "$pid1"
    wait "$pid2"
done

# nohup ./run_multiple_folders_on_test500DROP.sh &> run_multiple_folders_on_test500DROP.out &



#  # Wait for each command within the folder iteration to complete
# wait "$pid1"
