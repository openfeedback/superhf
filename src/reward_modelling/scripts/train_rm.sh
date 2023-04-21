#!/usr/bin/bash
#SBATCH --job-name=train_rm
#SBATCH --output=batch_outputs/test_job.%j.out
#SBATCH --error=batch_outputs/test_job.%j.err
#SBATCH --account=nlp
#SBATCH --mem=200GB
#SBATCH --cpus-per-gpu=16

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source ~/rm/bin/activate
export NCCL_P2P_DISABLE=1 # look at https://github.com/microsoft/DeepSpeed/issues/2176
export HF_DATASETS_CACHE="/nlp/scr/fongsu/.cache"
# deepspeed src/reward_modelling/reward_model.py --output_dir . --deepspeed src/reward_modelling/ds_configs/base_config.json
deepspeed src/reward_modelling/reward_model.py \
--model_name_or_path="EleutherAI/gpt-neo-1.3B" \
--output_dir="/nlp/src/fongsu/reward_model_HH/train_first_half/" \
--logging_steps=10 \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=16 \
--num_train_epochs=1 \
--do_eval=True \
--evaluation_strategy="steps" \
--eval_steps=1000 \
--save_total_limit=5 \
--save_strategy="steps" \
--save_steps=1000 \
--learning_rate=1e-5 \
--weight_decay=0.001 \
--report_to="wandb" \
--gradient_checkpointing=True \
--ddp_find_unused_parameters=False \
--push_to_hub=True \
--hub_model_id='gptneo-1.3B-rm-combined-train-first-half' \
--deepspeed src/reward_modelling/ds_configs/stage_3_config.json
