#!/bin/bash

# Set the number of days as a floating point variable
DAYS=4.0

# Calculate the number of iterations (hours)
iterations=$(awk "BEGIN {print $DAYS * 24}")

# Convert the float to an integer to use in the loop
iterations_int=$(printf "%.0f" $iterations)

# Loop for the calculated number of iterations
for i in $(seq 1 $iterations_int); do
  # Calculate the fractional days
  fractional_days=$(awk "BEGIN {print $i / 24}")

  # Print the progress
  echo "Syncing wandb experiments... (iteration $i/$iterations_int, day $fractional_days/$DAYS)"

  # Do the sync
  wandb sync wandb/run*

  # Sleep for 1 hour (3600 seconds)
  sleep 3600
done
