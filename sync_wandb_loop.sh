#!/bin/bash
# Calculate the number of iterations for X days (X days * 24 hours/day)
days = 4
iterations = $(bc -l <<< "$days * 24")

for i in $(seq 1 $iterations); do
  echo "Syncing wandb run $i of $iterations ($(bc -l <<< "$i / 24") of $days days)"
  sleep 1
  # wandb sync wandb/run*
  # sleep 3600  # Sleep for 1 hour (3600 seconds)
done
