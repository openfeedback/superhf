run_name=$2
file=$1
prompt_dict_path=$3


python "${file}" \
  --run_name "${run_name}" \
  --step_per_device_batch_size 1 \
  --rollout_per_device_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --output_dir "mocking_output_alpaca_farm" \
  --reward_model_name_or_path "peterchatain/mock_llama" \
  --policy_model_name_or_path "peterchatain/mock_llama" \
  --init_value_with_reward True \ # TODO: change to False
  --rollout_batch_size 8 \
  --step_batch_size 1 \
  --learning_rate 1e-5 \
  --warmup_steps 5 \
  --kl_coef 0.1 \
  --total_epochs 1 \
  --flash_attn False \
  --prompt_dict_path "${prompt_dict_path}" \
  --save_steps 99999 \
  --lora_r 4
