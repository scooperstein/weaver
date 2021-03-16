#!/bin/bash
if [ -z "$1" ]
  then
    echo "no jobname supplied"
    return
fi

export jobname=$1
echo $jobname
mkdir -p $jobname

cp -r evaluate.sh ../train.py ../data/AK4/ak4HLT_points_pf_sv_eta2p5.yaml ../data/AK4/ak4HLT_points_pf_sv_eta2p5_3classes.yaml ../networks/particle_net_pf_sv.py ../utils/ ../networks/particle_net_ak4_pf_sv.py evaluate.sub $jobname
cd $jobname
condor_submit evaluate.sub
cd ..


