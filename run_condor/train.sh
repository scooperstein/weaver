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
#cp /eos/cms/store/group/phys_exotica/monojet/rgerosa/TriggerHH/Ntuples_v2/tree_*.root .
#cp /eos/cms/store/group/phys_exotica/monojet/rgerosa/TriggerHH/Ntuples_v2/tree_QCD_Pt*.root .
#cp /eos/cms/store/group/phys_exotica/monojet/rgerosa/TriggerHH/Ntuples_v2/tree_ggHH*.root .
#cp /eos/cms/store/group/phys_exotica/monojet/rgerosa/TriggerHH/Ntuples_v2/tree_TT_TuneCP5_14TeV-powheg-pythia8.root .
##xrdcp --retry 2 --parallel 2 -r -f root://eoscms.cern.ch//eos/cms/store/group/phys_exotica/monojet/rgerosa/TriggerHH/Ntuples_v2/ .
#xrdcp --retry 2 --parallel 2 -r -f root://eoscms.cern.ch//eos/cms/store/group/phys_exotica/monojet/rgerosa/TriggerHH/NtuplesTraining/ .
cp /eos/cms/store/group/phys_exotica/monojet/rgerosa/TriggerHH/NtuplesTraining/tree_*.root .
echo "done copying files"
ls

echo "activating conda environment"
eval "$(conda shell.bash hook)"
conda activate /afs/cern.ch/user/s/scoopers/miniconda3/envs/weaver

#sleep 1000000
echo "running training"
##python train.py --data-train /eos/cms/store/group/phys_exotica/monojet/rgerosa/TriggerHH/Ntuples_v2/tree_*.root /eos/cms/store/group/phys_exotica/monojet/rgerosa/TriggerHH/Ntuples/tree_*thread1.root --data-config ak4HLT_points_pf_sv_eta2p5.yaml --network-config particle_net_pf_sv.py --model-prefix /eos/cms/store/user/scoopers/HH4b/TriggerModels/march15_moreSamplesTNV2_AK8PNet_RWT30percentile_deepjetClassWeights_onefileperprocess_2splitSix/ --num-workers 4 --gpus '0' --batch-size 512 --start-lr 5e-3 --num-epochs 20 --optimizer ranger 

python train.py --data-train tree_*.root --data-config ak4HLT_points_pf_sv_eta2p5.yaml --network-config particle_net_ak4_pf_sv.py --model-prefix /eos/cms/store/user/scoopers/HH4b/TriggerModels/march25_AK4PNet_HuilinYaml_nodeCopy/ --num-workers 4 --gpus '0' --batch-size 512 --start-lr 2e-2 --num-epochs 20 --optimizer ranger --fetch-step 0.005 --data-fraction 0.9 --train-val-split 0.9 --load-epoch 14 
echo "done training, removing root files..."

rm tree_*.root
echo "done"

