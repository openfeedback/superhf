#!/usr/bin/bash
#SBATCH --job-name=train_harmless
#SBATCH --output=batch_outputs/test_job.%j.out
#SBATCH --error=batch_outputs/test_job.%j.err
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=200GB
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source ~/rm/bin/activate
export NCCL_P2P_DISABLE=1 # look at https://github.com/microsoft/DeepSpeed/issues/2176
deepspeed --num_gpus=4 src/reward_modelling/reward_model.py --deepspeed src/reward_modelling/ds_configs/stage_3_config.json
