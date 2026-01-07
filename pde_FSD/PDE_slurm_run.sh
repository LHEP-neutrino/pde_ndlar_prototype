#!/bin/bash
#SBATCH --job-name=PDE_claculations_fsd
#SBATCH --account=dune
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=7
#SBATCH --ntasks-per-node=100
#SBATCH --time=02:00:00
#SBATCH --output=logs/pde_calculation_%j.out
#SBATCH --error=logs/pde_calculation_%j.err

mkdir -p logs

# activate the 
source ~/ndlar_flow.venv/bin/activate

#starting the scripts, the index of the files are given as: index = int(os.environ["SLURM_PROCID"])
# run this .sh script with with sbatch muon_selection_v2_debug.sh
# check on job: scontrol show job <jobid>
# real time monitoring with: watch -n 2 squeue -u $USER

# match the name of the .py file with 
srun python3 1_PDE_iterate_through_files_and_events.py