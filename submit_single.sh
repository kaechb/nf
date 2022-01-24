#!/bin/bash
#SBATCH --partition=allgpu
#SBATCH --constraint='V100'|'A100'
#SBATCH --time=78:00:00                           # Maximum time requested
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --chdir=/home/kaechben/slurm/output        # directory must already exist!
#SBATCH --job-name=hostname
#SBATCH --output=%j.out               # File to which STDOUT will be written
#SBATCH --error=%j.err                # File to which STDERR will be written
#SBATCH --mail-type=END                           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=max.muster@desy.de            # Email to which notifications will be sent. It defaults to <userid@mail.desy.de> if none is set.
unset LD_PRELOAD
source /etc/profile.d/modules.sh
module purge
module load maxwell
module load anaconda3/5.2
. conda-init
conda activate fixray
python -u /home/kaechben/JetNet_NF/nf_scan.py q

