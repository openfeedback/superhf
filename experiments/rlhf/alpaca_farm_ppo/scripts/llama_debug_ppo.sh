file=$1
run_name=$2
config_file=$3
prompt_dict_path=$4

root_path="experiments/rlhf/alpaca_farm_ppo"
# check if config file exists
# if it does, add config file to command
# if not, use default config file
if [ -f "${config_file}" ]; then
  echo "Using Accelerate config file: ${config_file}"
  command="accelerate launch --config_file ${config_file} ${root_path}/rlhf_v2.py"
else
  echo "Accelerate config file not provided. Using default config file."
  command="accelerate launch ${root_path}/rlhf_v2.py"
fi

#run command
${command} \
  --run_name "${run_name}" \
  --step_per_device_batch_size 1 \
  --rollout_per_device_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --output_dir "llama_output_alpaca_farm" \
  --reward_model_name_or_path "/juice5/scr5/nlp/llama_model/llama_hf_latest/llama-7b" \
  --policy_model_name_or_path "/juice5/scr5/nlp/llama_model/llama_hf_latest/llama-7b" \
  --init_value_with_reward True \ # TODO: Change to false
  --rollout_batch_size 8 \
  --step_batch_size 1 \
  --learning_rate 1e-5 \
  --warmup_steps 5 \
  --kl_coef 0.1 \
  --total_epochs 1 \
  --flash_attn False \
  --prompt_dict_path "${prompt_dict_path}" \
  --save_steps 99999 \
  --lora_r 4 \
