#!/bin/bash
#python3 -m virtualenv myvenv
#source myvenv/bin/activate
mkdir utils
mv data  dataset.py  __init__.py  logger.py  lr_finder.py  nn  __pycache__ utils

ls
ls utils

#sleep 5000000

#echo "setting up conda"
#conda init bash
## create a new conda environment
#conda create -n weaver python=3.7
#
## activate the environment
#conda activate weaver
#
#echo "installing packages"
## install the necessary python packages
#pip install numpy pandas scikit-learn scipy matplotlib tqdm PyYAML
#
## install uproot for reading/writing ROOT files
#pip install uproot==3.13.1 awkward==0.14.0 lz4 xxhash
#
## install PyTables if using HDF5 files
#pip install tables
#
## install onnxruntime if needs to run inference w/ ONNX models
#pip install onnxruntime-gpu
#
## install pytorch, follow instructions for your OS/CUDA version at:
## https://pytorch.org/get-started
#pip install torch
echo "copying files"
#cp /eos/cms/store/group/phys_higgs/HiggsExo/H2Mu/rgerosa/TriggerNtuplesReducedEval/tree_TT_TuneCP5_14TeV-powheg-pythia8_*.root .
#cp /eos/cms/store/group/phys_exotica/monojet/rgerosa/TriggerHH/NtuplesEvaluation/tree_QCD_Pt_170to300_TuneCP5_14TeV_pythia8_block0_thread*.root .
cp /eos/cms/store/group/phys_exotica/monojet/rgerosa/TriggerHH/NtuplesEvaluation/tree_QCD_Pt_50to80_TuneCP5_14TeV_pythia8_block0_thread*.root .
cp /eos/cms/store/user/scoopers/HH4b/TriggerModels/ak4HLT_huilin_apr8b.yaml . 
echo "done copying files"
ls

echo "activating conda environment"
eval "$(conda shell.bash hook)"
conda activate /afs/cern.ch/user/s/scoopers/miniconda3/envs/weaver

ulimit -s unlimited
echo "running training"
python train.py --predict --data-test 'tree_QCD*.root'  --data-config ak4HLT_huilin_apr8b.yaml  --network-config particle_net_pf_sv.py --model-prefix /eos/cms/store/user/scoopers/HH4b/TriggerModels/huilin_apr8b.pt  --num-workers 1 --gpus '0' --batch-size 512  --predict-output /eos/cms/store/group/phys_higgs/HiggsExo/H2Mu/rgerosa/TriggerNtuplesPNetHLTEval/tree_QCD_Pt_50to80_block0_april8b_huilin.root
##python train.py --predict --data-test 'tree_TT_TuneCP5_14TeV-powheg-pythia8_*.root'  --data-config ak4HLT_points_pf_sv_eta2p5.yaml  --network-config particle_net_ak4_pf_sv.py --model-prefix /eos/cms/store/user/scoopers/HH4b/TriggerModels/march25_AK4PNet_HuilinYaml_nodeCopy/_best_epoch_state.pt  --num-workers 1 --gpus '0' --batch-size 512  --predict-output /eos/cms/store/group/phys_higgs/HiggsExo/H2Mu/rgerosa/TriggerNtuplesPNetHLTEval/tree_TT_TuneCP5_14TeV-powheg-pythia8_march25_AK4PNet_HuilinYaml_nodeCopy_epoch13.root
