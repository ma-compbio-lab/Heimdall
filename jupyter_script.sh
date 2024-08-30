#!/bin/bash
#SBATCH -p ma2-gpu
#SBATCH -w compute-4-13
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=24GB
#SBATCH --job-name jupyter_notebook
#SBATCH --output jupyter-log-%J.txt
#SBATCH -t 2-0:00:00

# --> uncomment to use conda environment with jupyter installed <---
source activate scDRS

## get tunneling info
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)

## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -N -L $ipnport:$ipnip:$ipnport $(whoami)@lanec1.compbio.cs.cmu.edu
    -----------------------------------------------------------------
    Then open a browser on your local machine to the following address
    ------------------------------------------------------------------
    localhost:$ipnport  (prefix w/ https:// if using password)
    ------------------------------------------------------------------
    "
## start an ipcluster instance and launch jupyter server
jupyter-notebook --NotebookApp.iopub_data_rate_limit=100000000000000 --no-browser --port=$ipnport --ip=$ipnip --NotebookApp.password='' --NotebookApp.token=''
Running a jupyter notebook:
