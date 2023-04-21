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
deepspeed src/reward_modelling/reward_model.py --output_dir . 

