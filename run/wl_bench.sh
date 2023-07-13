#!/usr/bin/bash --login

# Global settings
NUM_REPS=10
INIT_SEED=1  # we can always extend the number of runs by keeping NUM_REPS=1 and then increment INIT_SEED
WRAPPER=local  # local, wrapper_msuicer, wrapper_mila
CONFIG_DIR=configs/wl_bench
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
    pse=$2

    run_script="python main.py --cfg ${CONFIG_DIR}/${dataset}-GIN+${pse}.yaml --repeat ${NUM_REPS} seed ${INIT_SEED} wandb.use ${USE_WANDB}"
    full_script="${job_script}${run_script}"

    echo $full_script  # print out the command
    eval $full_script  # execute the command
}

launch csl none
launch csl rand
launch csl LapPE
launch csl RWSE
launch csl GPSE

launch exp none
launch exp rand
launch exp LapPE
launch exp RWSE
launch exp GPSE
