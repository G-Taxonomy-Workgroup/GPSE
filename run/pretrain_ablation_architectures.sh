#!/usr/bin/bash --login

# Global settings
NUM_REPS=1
INIT_SEED=1  # we can always extend the number of runs by keeping NUM_REPS=1 and then increment INIT_SEED
WRAPPER=local  # local, wrapper_msuicer, wrapper_mila
CONFIG=configs/pretrain/gpse_molpcba_ablation_template.yaml
USE_WANDB=False
LAYERS=(
5
10
15
20
30
40
)
#################

HOME_DIR=$(dirname $(realpath $0))
ROOT_DIR=$(dirname $HOME_DIR)
echo HOME_DIR=$HOME_DIR
echo ROOT_DIR=$ROOT_DIR

cd $ROOT_DIR

run_script+="python main.py --cfg ${CONFIG} --repeat ${NUM_REPS} "
run_script+="seed ${INIT_SEED} wandb.use ${USE_WANDB} train.record_individual_scores True"

if [[ $WRAPPER != "local" ]]; then
    mkdir -p ${ROOT_DIR}/slurm_history
    run_script="sbatch -c 5 --mem=45GB -o ${ROOT_DIR}/slurm_history/slurm-%A.out run/${WRAPPER}.sb ${run_script}"
fi

launch () {
    command=$1
    echo $command  # print out the command
    eval $command  # execute the command
}

# Default settings
launch "${run_script} name_tag default"

# Conv layer type
# launch "${run_script} name_tag resgatedgcnconv gnn.layer_type resgatedgcnconv"  # (default)
launch "${run_script} name_tag gatv2conv gnn.layer_type gatv2conv"
launch "${run_script} name_tag gcnconv gnn.layer_type gcnconv"
launch "${run_script} name_tag ginconv gnn.layer_type ginconv"

# Independent MLP heads head vs. shared MLP head
# launch "${run_script} name_tag mlpindep gnn.head inductive_hybrid_multi"  # (default)
launch "${run_script} name_tag mlpshared gnn.head inductive_hybrid"

# Hidden dimension
launch "${run_script} name_tag diminner256 gnn.dim_inner 256"
# launch "${run_script} name_tag diminner512 gnn.dim_inner 512"  # (default)

# Convolution type + vn + layers
for layers in ${LAYERS[@]}; do
    # GatedGCN
    launch "${run_script} name_tag gatedgcn_novn_layers${layers} gnn.layer_type resgatedgcnconv gnn.layers_mp ${layers} virtual_node False"
    # GatedGCN (virtual node)
    launch "${run_script} name_tag gatedgcn_vn_layers${layers} gnn.layer_type resgatedgcnconv gnn.layers_mp ${layers} virtual_node True"
    # GIN (virtual node)
    launch "${run_script} name_tag gin_novn_layers${layers} gnn.layer_type ginconv gnn.layers_mp ${layers} virtual_node False"
    # GIN
    launch "${run_script} name_tag gin_vn_layers${layers} gnn.layer_type ginconv gnn.layers_mp ${layers} virtual_node True"
done
