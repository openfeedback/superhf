#SBATCH --error=batch_outputs/test_job.%j.err
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source ~/rm/bin/activate
# ./scripts/train_rm/run_gptneo.sh
deepspeed --num_gpus=1 src/reward_modelling/reward_model.py --deepspeed src/reward_modelling/ds_configs/base_configs.json
