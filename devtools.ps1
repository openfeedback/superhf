# Define the conda environment name
$CONDA_ENV = "superhf"

# Hint for how to use this script
function Usage {
    Write-Output "Usage: ./devtools.ps1 [command]"
    Write-Output "Commands:"
    Write-Output "  activate: Activate the conda environment or create it if it doesn't exist"
    Write-Output "  upgrade: Upgrade the conda environment and pip dependencies"
    Write-Output "  installgpu: Install pip dependencies and PyTorch (GPU)"
    Write-Output "  installcpu: Install pip dependencies and PyTorch (CPU)"
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
    Invoke-Expression "pip-compile --upgrade -v requirements/prod.in"
    Invoke-Expression "pip-compile --upgrade -v requirements/dev.in"
    Invoke-Expression "pip-compile --upgrade -v requirements/torchgpu.in"
    Invoke-Expression "pip-compile --upgrade -v requirements/torchcpu.in"
}

# Install frozen pip packages and PyTorch (GPU)
function Install-GPU {
    Activate
    Invoke-Expression "pip-sync requirements/prod.txt requirements/dev.txt requirements/torchgpu.txt"
    Invoke-Expression "pip install -e ."
    Invoke-Expression "pre-commit install"
}

# Install frozen pip packages and PyTorch (CPU)
function Install-CPU {
    Activate
    Invoke-Expression "pip-sync requirements/prod.txt requirements/dev.txt requirements/torchcpu.txt"
    Invoke-Expression "pip install -e ."
    Invoke-Expression "pre-commit install"
}

# Workflows
switch ($args[0]) {
    activate {
        Activate
    }
    upgradegpu {
        Upgrade-Requirements
        Install-GPU
    }
    upgradecpu {
        Upgrade-Requirements
        Install-CPU
    }
    installgpu {
        Install-GPU
    }
    installcpu {
        Install-CPU
    }
    default {
        Usage
    }
}
