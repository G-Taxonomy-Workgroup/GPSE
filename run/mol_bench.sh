#!/usr/bin/bash --login

# Global settings
NUM_REPS=10
INIT_SEED=1  # we can always extend the number of runs by keeping NUM_REPS=1 and then increment INIT_SEED
WRAPPER=local  # local, wrapper_msuicer, wrapper_mila
CONFIG_DIR=configs/mol_bench
USE_WANDB=False
#################

HOME_DIR=$(dirname $(realpath $0))
ROOT_DIR=$(dirname $HOME_DIR)
echo HOME_DIR=$HOME_DIR
echo ROOT_DIR=$ROOT_DIR

cd $ROOT_DIR

if [[ $WRAPPER != "local" ]]; then
    mkdir -p ${ROOT_DIR}/slurm_history
    job_script="sbatch -c 5 --mem=45GB -o ${ROOT_DIR}/slurm_history/slurm-%A.out run/${WRAPPER}.sb "
fi

launch () {
    dataset=$1
    model=$2
    pse=$3

    name="${dataset}-${model}+${pse}"
    run_script="python main.py --cfg ${CONFIG_DIR}/${name}.yaml --repeat ${NUM_REPS} seed ${INIT_SEED} wandb.use ${USE_WANDB}"
    full_script="${job_script}${run_script}"

    echo $full_script  # print out the command
    eval $full_script  # execute the command
}

# ZINC
launch zinc GPS none
launch zinc GPS rand
launch zinc GPS LapPE
launch zinc GPS RWSE
launch zinc GPS GPSE

launch zinc GIN none
launch zinc GIN rand
launch zinc GIN LapPE
launch zinc GIN RWSE
launch zinc GIN GPSE

launch zinc GCN none
launch zinc GCN rand
launch zinc GCN LapPE
launch zinc GCN RWSE
launch zinc GCN GPSE

launch zinc GatedGCN none
launch zinc GatedGCN rand
launch zinc GatedGCN LapPE
launch zinc GatedGCN RWSE
launch zinc GatedGCN GPSE

launch zinc Transformer none
launch zinc Transformer rand
launch zinc Transformer LapPE
launch zinc Transformer RWSE
launch zinc Transformer GPSE

# PCQM4Mv2-subset
launch pcqm4msubset GPS none
launch pcqm4msubset GPS rand
launch pcqm4msubset GPS LapPE
launch pcqm4msubset GPS RWSE
launch pcqm4msubset GPS GPSE

# MolHIV
launch molhiv GPS none
launch molhiv GPS rand
launch molhiv GPS LapPE
launch molhiv GPS RWSE
launch molhiv GPS GPSE

# MolPCBA
launch molpcba GPS none
launch molpcba GPS rand
launch molpcba GPS LapPE
launch molpcba GPS RWSE
launch molpcba GPS GPSE
