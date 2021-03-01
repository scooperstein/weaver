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

echo "activating conda environment"
eval "$(conda shell.bash hook)"
conda activate /afs/cern.ch/user/s/scoopers/miniconda3/envs/weaver

echo "running training"
python train.py --data-train '/eos/cms/store/group/phys_higgs/HiggsExo/H2Mu/rgerosa/TriggerNtuples/tree_TT_TuneCP5_14TeV-powheg-pythia8_*.root' --data-config ak4HLT_points_pf_sv_eta2p5.yaml --network-config particle_net_pf_sv.py --model-prefix /eos/cms/store/group/phys_higgs/HiggsExo/H2Mu/rgerosa/TriggerModels/ --num-workers 3 --gpus 0 --batch-size 512 --start-lr 5e-3 --num-epochs 20 --optimizer ranger 
