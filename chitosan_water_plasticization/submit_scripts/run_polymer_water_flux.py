#!/bin/bash
#SBATCH --job-name=eo4_1       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=50               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=05:00:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2021.11
conda activate openmm

cd "/scratch/gpfs/bu9134/plasticization/1/1"
python /projects/WEBB/eser/analysis_scripts/plasticization_analysis/water_flux.py

cd "/scratch/gpfs/bu9134/plasticization/1/2"
python /projects/WEBB/eser/analysis_scripts/plasticization_analysis/water_flux.py

cd "/scratch/gpfs/bu9134/plasticization/1/3"
python /projects/WEBB/eser/analysis_scripts/plasticization_analysis/water_flux.py

cd "/scratch/gpfs/bu9134/plasticization/1/4"
python /projects/WEBB/eser/analysis_scripts/plasticization_analysis/water_flux.py
