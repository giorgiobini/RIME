#!/bin/bash

folders=('arch2_PSORALENtrained_PARISval0005' 'arch2_PSORALENtrained_PARISval0006' 'arch2_PSORALENtrained_PARISval0007' 'arch2_PSORALENtrained_PARISval0008' 'arch2_PSORALENtrained_PARISval0009' 'arch2_PSORALENtrained_PARISval0010' 'arch2_PSORALENtrained_PARISval0011' 'arch2_PSORALENtrained_PARISval0012' 'arch2_PSORALENtrained_PARISval0013' 'arch2_PSORALENtrained_PARISval0014' 'arch2_PSORALENtrained_PARISval0015' 'arch2_PSORALENtrained_PARISval0016' 'arch2_PSORALENtrained_PARISval0017' 'arch2_PSORALENtrained_PARISval0018' 'arch2_PSORALENtrained_PARISval0019' 'arch2_PSORALENtrained_PARISval0020' 'arch2_PSORALENtrained_PARISval0021' 'arch2_PSORALENtrained_PARISval0022' 'arch2_PSORALENtrained_PARISval0023' 'arch2_PSORALENtrained_PARISval0024' 'arch2_PSORALENtrained_PARISval0025' 'arch2_PSORALENtrained_PARISval0026' 'arch2_PSORALENtrained_PARISval0027' 'arch2_PSORALENtrained_PARISval0028' 'arch2_PSORALENtrained_PARISval0029' 'arch2_PSORALENtrained_PARISval0030' 'arch2_PSORALENtrained_PARISval0031' 'arch2_PSORALENtrained_PARISval0032' 'arch2_PSORALENtrained_PARISval0033' 'arch2_PSORALENtrained_PARISval0034' 'arch2_PSORALENtrained_PARISval0035' 'arch2_PSORALENtrained_PARISval0036' 'arch2_PSORALENtrained_PARISval0037' 'arch2_PSORALENtrained_PARISval0038' 'arch2_PSORALENtrained_PARISval0039' 'arch2_PSORALENtrained_PARISval0040' 'arch2_PSORALENtrained_PARISval0041' 'arch2_PSORALENtrained_PARISval0042' 'arch2_PSORALENtrained_PARISval0043' 'arch2_PSORALENtrained_PARISval0044' 'arch2_PSORALENtrained_PARISval0045' 'arch2_PSORALENtrained_PARISval0046' 'arch2_PSORALENtrained_PARISval0047' 'arch2_PSORALENtrained_PARISval0048' 'arch2_PSORALENtrained_PARISval0049' 'arch2_PSORALENtrained_PARISval0050' 'arch2_PSORALENtrained_PARISval0051' 'arch2_PSORALENtrained_PARISval0052' 'arch2_PSORALENtrained_PARISval0053' 'arch2_PSORALENtrained_PARISval0054' 'arch2_PSORALENtrained_PARISval0055' 'arch2_PSORALENtrained_PARISval0056' 'arch2_PSORALENtrained_PARISval0057' 'arch2_PSORALENtrained_PARISval0058' 'arch2_PSORALENtrained_PARISval0059' 'arch2_PSORALENtrained_PARISval0060' 'arch2_PSORALENtrained_PARISval0061' 'arch2_PSORALENtrained_PARISval0062' 'arch2_PSORALENtrained_PARISval0063' 'arch2_PSORALENtrained_PARISval0064' 'arch2_PSORALENtrained_PARISval0065' 'arch2_PSORALENtrained_PARISval0066' 'arch2_PSORALENtrained_PARISval0067' 'arch2_PSORALENtrained_PARISval0068' 'arch2_PSORALENtrained_PARISval0069' 'arch2_PSORALENtrained_PARISval0070' 'arch2_PSORALENtrained_PARISval0071' 'arch2_PSORALENtrained_PARISval0072' 'arch2_PSORALENtrained_PARISval0074' 'arch2_PSORALENtrained_PARISval0075' 'arch2_PSORALENtrained_PARISval0076' 'arch2_PSORALENtrained_PARISval0077' 'arch2_PSORALENtrained_PARISval0078' 'arch2_PSORALENtrained_PARISval0079' 'arch2_PSORALENtrained_PARISval0081' 'arch2_PSORALENtrained_PARISval0082' 'arch2_PSORALENtrained_PARISval0083' 'arch2_PSORALENtrained_PARISval0084' 'arch2_PSORALENtrained_PARISval0085' 'arch2_PSORALENtrained_PARISval0086' 'arch2_PSORALENtrained_PARISval0088' 'arch2_PSORALENtrained_PARISval0089' 'arch2_PSORALENtrained_PARISval0090' 'arch2_PSORALENtrained_PARISval0091' 'arch2_PSORALENtrained_PARISval0093' 'arch2_PSORALENtrained_PARISval0095' 'arch2_PSORALENtrained_PARISval0096' 'arch2_PSORALENtrained_PARISval0097' 'arch2_PSORALENtrained_PARISval0098' 'arch2_PSORALENtrained_PARISval0099' 'arch2_PSORALENtrained_PARISval0103')

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

    nohup python run_binary_cl_on_test_500.py --folder="$folder" --how=val &> run_binary_cl_on_val.out &
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
