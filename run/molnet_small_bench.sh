#!/usr/bin/bash --login

# Global settings
NUM_REPS=10
INIT_SEED=1  # we can always extend the number of runs by keeping NUM_REPS=1 and then increment INIT_SEED
WRAPPER=local  # local, wrapper_msuicer, wrapper_mila
CONFIG=configs/molnet-GINE+GPSE.yaml
USE_WANDB=False
#################

HOME_DIR=$(dirname $(realpath $0))
ROOT_DIR=$(dirname $HOME_DIR)
echo HOME_DIR=$HOME_DIR
echo ROOT_DIR=$ROOT_DIR

cd $ROOT_DIR

run_script="python main.py --cfg ${CONFIG} --repeat ${NUM_REPS} seed ${INIT_SEED} wandb.use ${USE_WANDB}"

if [[ $WRAPPER != "local" ]]; then
    mkdir -p ${ROOT_DIR}/slurm_history
    job_script="sbatch -c 5 --mem=45GB -o ${ROOT_DIR}/slurm_history/slurm-%A.out run/${WRAPPER}.sb "
fi

launch () {
    dataset=$1
    task_type=$2

    full_script="${job_script}${run_script} dataset.name ${dataset} dataset.task_type ${task_type}"

    echo $full_script  # print out the command
    eval $full_script  # execute the command
}

launch ogbg-molbbbp classification
launch ogbg-molbace classification
launch ogbg-molsider multilabel
launch ogbg-moltox21 multilabel
launch ogbg-moltoxcast multilabel
