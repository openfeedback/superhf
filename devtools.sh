# Define the conda environment name
CONDA_ENV="superhf"

# Hint for how to use this script
function usage {
    echo "Usage: source devtools.sh [command]"
    echo "Commands:"
    echo "  activate: Activate the conda environment or create it if it doesn't exist"
    echo "  upgradegpu: Upgrade the dependencies and install PyTorch (GPU)"
    echo "  upgradecpu: Upgrade the dependencies and install PyTorch (CPU)"
    echo "  installgpu: Install pip dependencies and PyTorch (GPU)"
    echo "  installcpu: Install pip dependencies and PyTorch (CPU)"
    echo "  merge: Pull main and merge into this branch (no fast-forward)"
}

# Activate the conda environment or create it if it doesn't exist
function activate {
    # Create the environment if it doesn't exist
    envList=$(conda env list)
    envExists=$(echo "$envList" | grep -c "$CONDA_ENV")
    if [ "$envExists" -eq 0 ]; then
        echo "Conda environment '$CONDA_ENV' does not exist. Creating it now..."
        conda env create -f environment.yml
    fi
    echo "Activating conda environment '$CONDA_ENV'..."
    conda activate "$CONDA_ENV"
}

# Update dependencies to latest valid versions
function upgrade_requirements {
    activate

    # Update the conda environment
    conda env update --prune

    # Update the pip dependencies
    pip-compile --upgrade -v requirements/prod.in
    pip-compile --upgrade -v requirements/dev.in
    pip-compile --upgrade -v requirements/torchgpu.in
    pip-compile --upgrade -v requirements/torchcpu.in
}

# Install frozen pip packages and PyTorch (GPU)
function install_gpu {
    activate
    pip-sync requirements/prod.txt requirements/dev.txt requirements/torchgpu.txt
    pip install -e .
    pre-commit install
}

# Install frozen pip packages and PyTorch (CPU)
function install_cpu {
    activate
    pip-sync requirements/prod.txt requirements/dev.txt requirements/torchcpu.txt
    pip install -e .
    pre-commit install
}

# Pull main and merge into this branch (no fast-forward)
function pull_main {
    currentBranch=$(git rev-parse --abbrev-ref HEAD)
    git checkout main
    git pull
    git checkout "$currentBranch"
    git merge main --no-ff
}

# Workflows
case "$1" in
    activate)
        activate
        ;;
    upgradegpu)
        upgrade_requirements
        install_gpu
        ;;
    upgradecpu)
        upgrade_requirements
        install_cpu
        ;;
    installgpu)
        install_gpu
        ;;
    installcpu)
        install_cpu
        ;;
    merge)
        pull_main
        ;;
    *)
        usage
        ;;
esac
