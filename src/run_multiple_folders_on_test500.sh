#!/bin/bash

folders=('all_modelarch2_wrongSN_easypretrain11_NOWENHNokSN_splashPARISFINETUNINGtrainhqRICSEQval0015' 'all_modelarch2_wrongSN_easypretrain11_NOWENHNokSN_splashPARISFINETUNINGtrainhqRICSEQval0021' 'all_modelarch2_wrongSN_easypretrain11_NOWENHNokSN_splashPARISFINETUNINGtrainhqRICSEQval0022' 'all_modelarch2_wrongSN_easypretrain11_NOWENHNokSN_splashPARISFINETUNINGtrainhqRICSEQval0024' 'all_modelarch2_wrongSN_easypretrain11_NOWENHNokSN_splashPARISFINETUNINGtrainhqRICSEQval0026' 'all_modelarch2_wrongSN_easypretrain11_NOWENHNokSN_splashPARISFINETUNINGtrainhqRICSEQval0027' 'all_modelarch2_wrongSN_easypretrain11_NOWENHNokSN_splashPARISFINETUNINGtrainhqRICSEQval0034' 'all_modelarch2_wrongSN_easypretrain11_NOWENHNokSN_splashPARISFINETUNINGtrainhqRICSEQval0039' 'all_modelarch2_wrongSN_easypretrain11_NOWENHNokSN_splashPARISFINETUNINGtrainhqRICSEQval0059' 'all_modelarch2_wrongSN_easypretrain11_NOWENHNokSN_splashPARISFINETUNINGtrainhqRICSEQval0071' 'all_modelarch2_wrongSN_easypretrain11_paristfinetuningSPLASHval0039_NOWENHNokSN_PARISFINETUNINGtrainhqSPLASHval0043' 'all_modelarch2_wrongSN_easypretrain11_paristfinetuningSPLASHval0039_NOWENHNokSN_PARISFINETUNINGtrainhqSPLASHval0046' 'all_modelarch2_wrongSN_easypretrain11_paristfinetuningSPLASHval0039_NOWENHNokSN_PARISFINETUNINGtrainhqSPLASHval0047' 'all_modelarch2_wrongSN_easypretrain11_paristfinetuningSPLASHval0039_NOWENHNokSN_PARISFINETUNINGtrainhqSPLASHval0048' 'all_modelarch2_wrongSN_easypretrain11_paristfinetuningSPLASHval0039_NOWENHNokSN_PARISFINETUNINGtrainhqSPLASHval0049' 'all_modelarch2_wrongSN_easypretrain11_paristfinetuningSPLASHval0039_NOWENHNokSN_PARISFINETUNINGtrainhqSPLASHval0051' 'all_modelarch2_wrongSN_easypretrain11_paristfinetuningSPLASHval0039_NOWENHNokSN_PARISFINETUNINGtrainhqSPLASHval0053' 'all_modelarch2_wrongSN_easypretrain11_paristfinetuningSPLASHval0039_NOWENHNokSN_PARISFINETUNINGtrainhqSPLASHval0054' 'all_modelarch2_wrongSN_easypretrain11_paristfinetuningSPLASHval0039_NOWENHNokSN_PARISFINETUNINGtrainhqSPLASHval0055' 'all_modelarch2_wrongSN_easypretrain11_paristfinetuningSPLASHval0039_NOWENHNokSN_PARISFINETUNINGtrainhqSPLASHval0056' 'all_modelarch2_wrongSN_easypretrain11_paristfinetuningSPLASHval0039_NOWENHNokSN_PARISFINETUNINGtrainhqSPLASHval0062' 'all_modelarch2_wrongSN_easypretrain11_paristfinetuningSPLASHval0039_NOWENHNokSN_PARISFINETUNINGtrainhqSPLASHval0093')

for folder in "${folders[@]}"; do

    nohup python run_binary_cl_on_test_500.py --folder="$folder" --dataset=splash &> run_binary_cl_on_test_splash500.out &
    pid1=$!

    nohup python run_binary_cl_on_test_500.py --folder="$folder" --dataset=splash --enhn &> run_binary_cl_on_test_splashENHN500.out &
    pid2=$!

    # Wait for each command within the folder iteration to complete
    wait "$pid1"
    wait "$pid2"


    nohup python run_binary_cl_on_test_500.py --folder="$folder" &> run_binary_cl_on_test_500.out &
    pid3=$!

    nohup python run_binary_cl_on_test_500.py --folder="$folder" --dataset=mario &> run_binary_cl_on_test_mario500.out &
    pid4=$!

    nohup python run_binary_cl_on_test_500.py --folder="$folder" --dataset=ricseq &> run_binary_cl_on_test_ricseq500.out &
    pid5=$!

    nohup python run_binary_cl_on_test_500.py --folder="$folder" --enhn &> run_binary_cl_on_testENHN500.out &
    pid6=$!

    nohup python run_binary_cl_on_test_500.py --folder="$folder" --dataset=mario --enhn &> run_binary_cl_on_test_marioENHN500.out &
    pid7=$!

    nohup python run_binary_cl_on_test_500.py --folder="$folder" --dataset=ricseq --enhn &> run_binary_cl_on_test_ricseqENHN500.out &
    pid8=$!

    wait "$pid3"
    wait "$pid4"
    wait "$pid5"
    wait "$pid6"
    wait "$pid7"
    wait "$pid8"
done

# nohup ./run_multiple_folders_on_test500.sh &> run_multiple_folders_on_test500.out &
