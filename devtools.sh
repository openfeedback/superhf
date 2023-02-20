# Define the conda environment name
CONDA_ENV="superhf"

# Hint for how to use this script
function usage {
    echo "Usage: source devtools.sh [command]"
    echo "Commands:"
    echo "  activate: Activate the conda environment or create it if it doesn't exist"
    echo "  upgrade: Upgrade the dependencies, freeze them, and install them"
    echo "  install: Just install the frozen dependencies"
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
}

# Install frozen pip packages
function install {
    activate
    pip-sync requirements/prod.txt requirements/dev.txt
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
    upgrade)
        upgrade_requirements
        install
        ;;
    install)
        install
        ;;
    merge)
        pull_main
        ;;
    *)
        usage
        ;;
esac
