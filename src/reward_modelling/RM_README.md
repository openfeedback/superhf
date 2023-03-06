# How to train with deepspeed
Run the following command under the root directory of the repo.
```shell
deepspeed --num_gpus=1 src/reward_modelling/reward_model.py --deepspeed src/reward_modelling/ds_configs/base_configs.jsonhttps://github.com/openfeedback/superhf.git
```