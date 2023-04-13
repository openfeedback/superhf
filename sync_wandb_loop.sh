#!/bin/bash
for i in {1..16}; do
  wandb sync wandb/run*
  sleep 1800
done
