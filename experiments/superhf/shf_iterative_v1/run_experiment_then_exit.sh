#!/bin/bash
# Run the iterative SuperHF experiment and exit when it's done.

python run_shf_iterative.py --config configs/gpt-neo-125M.yaml --exit_on_completion
exit
