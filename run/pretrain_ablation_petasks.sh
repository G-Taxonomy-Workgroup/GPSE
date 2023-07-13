#!/usr/bin/bash --login

# Global settings
NUM_REPS=1
INIT_SEED=1  # we can always extend the number of runs by keeping NUM_REPS=1 and then increment INIT_SEED
WRAPPER=local  # local, wrapper_msuicer, wrapper_mila
CONFIG=configs/pretrain/gpse_molpcba_ablation_template.yaml
USE_WANDB=False
#################

HOME_DIR=$(dirname $(realpath $0))
ROOT_DIR=$(dirname $HOME_DIR)
echo HOME_DIR=$HOME_DIR
echo ROOT_DIR=$ROOT_DIR

cd $ROOT_DIR

run_script+="python main.py --cfg ${CONFIG} --repeat ${NUM_REPS} "
run_script+="seed ${INIT_SEED} wandb.use ${USE_WANDB} train.record_individual_scores True"
run_script+=" dataset.umg_train_ratio 0.9"  # full training

if [[ $WRAPPER != "local" ]]; then
    mkdir -p ${ROOT_DIR}/slurm_history
    run_script="sbatch -c 5 --mem=45GB -o ${ROOT_DIR}/slurm_history/slurm-%A.out run/${WRAPPER}.sb ${run_script}"
fi

launch () {
    command=$1
    echo $command  # print out the command
    eval $command  # execute the command
}

# Exclude electrostatic encoders
launch "${run_script} name_tag petask_noels dataset.output_node_encoders LapPE+RWSE+HKdiagSE posenc_ElstaticPE.enable False"

# Exclude Laplacian eigenvectors and eigenvalues
launch "${run_script} name_tag petask_noeig dataset.output_node_encoders ElstaticPE+RWSE+HKdiagSE dataset.output_graph_encoders CycleGE posenc_LapPE.enable False"

# Exclude random walk encoders
launch "${run_script} name_tag petask_norw dataset.output_node_encoders ElstaticPE+LapPE+HKdiagSE posenc_RWSE.enable False"

# Exclude heat kernel encoders
launch "${run_script} name_tag petask_nohk dataset.output_node_encoders ElstaticPE+LapPE+RWSE posenc_HKdiagSE.enable False"

# Exclude cycle counting encoders
launch "${run_script} name_tag petask_nocyc dataset.output_graph_encoders EigVals graphenc_CycleGE.enable False"
