#!/bin/bash

folders=('all_modelarch2_easypretrain11_parisANDsplashRICSEQval0016' 'all_modelarch2_easypretrain11_parisANDsplashRICSEQval0020' 'all_modelarch2_easypretrain11_parisANDsplashRICSEQval0030' 'all_modelarch2_easypretrain11_parisANDsplashRICSEQval0033' 'all_modelarch2_easypretrain11_parisANDsplashRICSEQval0062' 'all_modelarch2_easypretrain11_parisANDsplashRICSEQval0065' 'all_modelarch2_easypretrain11_parisANDsplashRICSEQval0071' 'all_modelarch2_easypretrain11_parisANDsplashRICSEQval0078' 'all_modelarch2_easypretrain11_parisANDsplashRICSEQval0081' 'all_modelarch2_easypretrain11_parisANDsplashRICSEQval0083' 'all_modelarch2_easypretrain11_parisANDsplashRICSEQval0087' 'all_modelarch2_easypretrain11_parisANDsplashRICSEQval0094' 'all_modelarch2_easypretrain11_parisANDsplashRICSEQval0099' 'all_modelarch2_easypretrain11_parisANDsplashRICSEQval0106' 'all_modelarch2_easypretrain11_parisANDsplashRICSEQval0103' 'all_modelarch2_easypretrain11_parisANDsplashRICSEQval0110' 'all_modelarch2_easypretrain11_parisANDsplashRICSEQval0111' 'all_modelarch2_easypretrain11_parisANDsplashRICSEQval0132' 'all_modelarch2_easypretrain11_parisANDsplashRICSEQval0136' 'all_modelarch2_easypretrain11_parisANDsplashRICSEQval0147')

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
