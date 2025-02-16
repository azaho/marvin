#!/bin/bash
#SBATCH --job-name=hand_to_neuro_run          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=4    # Request 8 CPU cores per GPU
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH --constraint=high-capacity
#SBATCH -t 12:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=0-143      # 14 jobs (108/8 rounded up)
#SBATCH --output r/%A_%a.out # STDOUT
#SBATCH --error r/%A_%a.err # STDERR

source .venv/bin/activate
# Create arrays for each parameter
d_models=(512 1024 2048)
latent_dims=(-1 64) 
model_types=(lstm transformer)
learning_rates=(0.001 0.0005 0.0001)
weight_decays=(0 0.0001)

# Get parameters for this array job
idx=$SLURM_ARRAY_TASK_ID

# Calculate indices for each parameter
d_model_idx=$((idx % ${#d_models[@]}))
idx=$((idx / ${#d_models[@]}))
latent_dim_idx=$((idx % ${#latent_dims[@]}))
idx=$((idx / ${#latent_dims[@]}))
model_type_idx=$((idx % ${#model_types[@]}))
idx=$((idx / ${#model_types[@]}))
lr_idx=$((idx % ${#learning_rates[@]}))
idx=$((idx / ${#learning_rates[@]}))
wd_idx=$((idx % ${#weight_decays[@]}))

# Get parameter values
d_model=${d_models[$d_model_idx]}
latent_dim=${latent_dims[$latent_dim_idx]}
model_type=${model_types[$model_type_idx]}
lr=${learning_rates[$lr_idx]}
weight_decay=${weight_decays[$wd_idx]}

echo "d_model: $d_model"
echo "latent_dim: $latent_dim"
echo "model_type: $model_type"
echo "lr: $lr"
echo "weight_decay: $weight_decay"
echo ""

# Run training with selected parameters
python -u hand_to_neuro_train.py --d_model $d_model --latent_dim $latent_dim --model_type $model_type --lr $lr --weight_decay $weight_decay