#!/usr/bin/bash --login
# Submit MoleculeNet benchmark jobs
#
# Examples:
#   $ PSE_MODE=GPSE bash molnet_bench.sh  # run using GPSE augmentation (default)
#   $ PSE_MODE=AllPSE bash molnet_bench.sh  # run using AllPSE augmentation
#   $ PSE_MODE=GraphLog bash molnet_bench.sh  # run using GraphLog augmentation

# Global settings
NUM_REPS=10
INIT_SEED=1  # we can always extend the number of runs by keeping NUM_REPS=1 and then increment INIT_SEED
WRAPPER=local  # local, wrapper_msuicer, wrapper_mila
CONFIG_DIR=configs/molnet_bench
USE_WANDB=False
PSE_MODE=GPSE
#################

HOME_DIR=$(dirname $(realpath $0))
ROOT_DIR=$(dirname $HOME_DIR)
echo HOME_DIR=$HOME_DIR
echo ROOT_DIR=$ROOT_DIR

cd $ROOT_DIR

if [[ $WRAPPER != "local" ]]; then
    mkdir -p ${ROOT_DIR}/slurm_history
    job_script="sbatch -c 5 --mem=45GB -o ${ROOT_DIR}/slurm_history/slurm-%A.out run/${WRAPPER}.sb"
fi

launch () {
    dataset=$1

    config=${CONFIG_DIR}/${dataset}-GINE+${PSE_MODE}.yaml
    run_script="python main.py --cfg ${config} --repeat ${NUM_REPS} seed ${INIT_SEED} wandb.use ${USE_WANDB}"
    full_script="${job_script} ${run_script}"

    echo $full_script  # print out the command
    eval $full_script  # execute the command
}

launch bbbp  # classification
launch bace  # classification
launch sider  # multilabel
launch tox21  # multilabel
launch toxcast  # multilabel
launch clintox  # multilabel
launch hiv  # classification
launch muv  # multilabel
