#!/bin/bash
#SBATCH --job-name=PDE_calculations_fsd
#SBATCH --account=dune
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=50
#SBATCH --time=00:15:00
#SBATCH --output=logs/pde_calculation_%j.out
#SBATCH --error=logs/pde_calculation_%j.err

mkdir -p logs

# activate the 
module load python
source /global/cfs/cdirs/dune/users/mnuland/run2flow/ndlar_flow.venv/bin/activate

#starting the scripts, the index of the files are given as: index = int(os.environ["SLURM_PROCID"])
# run this .sh script with with sbatch muon_selection_v2_debug.sh
# check on job: scontrol show job <jobid>
# real time monitoring with: watch -n 2 squeue -u $USER

# match the name of the .py file with 
srun python3 1_PDE_iterate_through_files_and_events.py