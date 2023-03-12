#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28000MB
#SBATCH --account=tra23_HpcAiQc
#SBATCH --partition=g100_usr_prod
#SBATCH --time=00:05:00
#SBATCH --error=job.%j.err
#SBATCH --output=job.%j.out
#SBATCH --job-name=nome_job


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load autoload
python ibm.py
