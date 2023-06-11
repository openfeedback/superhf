file=$1
run_name=$2
prompt_dict_path=$3

root_path="experiments/rlhf/alpaca_farm_ppo"
config_file="${root_path}/accelerate_configs/rlhf_ppo_fsdp_llama_1gpu.yaml"

accelerate launch --config_file "${config_file}" "${root_path}/rlhf_v2.py"\
  --run_name "${run_name}" \
  --step_per_device_batch_size 1 \
  --rollout_per_device_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --output_dir "llama_output_alpaca_farm" \
  --reward_model_name_or_path "/juice5/scr5/nlp/llama_model/llama_hf_latest/llama-7b" \
  --policy_model_name_or_path "/juice5/scr5/nlp/llama_model/llama_hf_latest/llama-7b" \
  --init_value_with_reward False \
  --rollout_batch_size 1 \
  --step_batch_size 1 \
  --learning_rate 1e-5 \
  --warmup_steps 5 \
  --kl_coef 0.1 \
  --total_epochs 1 \
  --flash_attn False \
  --prompt_dict_path "${prompt_dict_path}" \
  --save_steps 20
