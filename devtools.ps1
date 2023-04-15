# Define the conda environment name
$CONDA_ENV = "superhf"

# Hint for how to use this script
function Usage {
    Write-Output "Usage: ./devtools.ps1 [command]"
    Write-Output "Commands:"
    Write-Output "  activate: Activate the conda environment or create it if it doesn't exist"
    Write-Output "  upgrade: Upgrade the dependencies and install PyTorch"
    Write-Output "  install: Install pip dependencies and PyTorch"
    Write-Output "  mergemain: Pull main and merge into this branch (no fast-forward)"
}

# Activate the conda environment or create it if it doesn't exist
function Activate {
    # Create the environment if it doesn't exist
    $envList = conda env list
    $envExists = $envList | Select-String -SimpleMatch $CONDA_ENV
    if (!$envExists) {
        Write-Output "Conda environment '$CONDA_ENV' does not exist. Creating it now..."
        Invoke-Expression "conda env create -f environment.yml"
    }
    Invoke-Expression "conda activate $CONDA_ENV"
}

# Update dependencies to latest valid versions
function Upgrade-Requirements {
    Activate

    # Update the conda environment
    "conda env update --prune"

    # Update the pip dependencies
    # TODO use --resolver=backtracking once https://github.com/jazzband/pip-tools/pull/1808 is merged
    Invoke-Expression "pip-compile --upgrade -v requirements/prod.in"
    Invoke-Expression "pip-compile --upgrade -v requirements/dev.in"
}

# Install frozen pip packages and PyTorch
function Install {
    Activate
    Invoke-Expression "pip-sync requirements/prod.txt requirements/dev.txt"
    # Force upgrade to CUDA version of PyTorch
    Invoke-Expression "pip install --upgrade torch --extra-index-url https://download.pytorch.org/whl/cu118 --user"
    Invoke-Expression "pip install -e ."
    Invoke-Expression "pre-commit install"
}

# Pull main and merge into this branch (no fast-forward)
function Merge-Into-Current-Branch {
    $currentBranch = git rev-parse --abbrev-ref HEAD
    git checkout main
    git pull
    git checkout $currentBranch
    git merge --no-ff main
}

# Workflows
switch ($args[0]) {
    activate {
        Activate
    }
    upgrade {
        Upgrade-Requirements
        Install
    }
    install {
        Install
    }
    merge {
        Merge-Into-Current-Branch
    }
    default {
        Usage
    }
}
