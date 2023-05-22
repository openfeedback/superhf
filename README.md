# SuperHF

This project is a research prototype for the paper `Supervised Iterative Fine-Tuning From Human Preferences`

## Directory structure

- `src/superhf/` contains the code for the SuperHFTrainer class and superhf code.
- `src/reward_modeling/` contains the code for training the reward model.
- `experiments/` contains the code for calling the trainers and running various experiments reported in the paper.
- `experiments/superhf/superhf_iterative_v1` contains the code for running the superhf experiments that were used.
- `experiments/rlhf/rlhf_v1` contains the code for running the rlhf experiments.
- `experiments/evaluations/` contains the code for evaluating the trained models.


## Installation

### Python package
Install the library with pip:
```bash
pip install superhf
```
