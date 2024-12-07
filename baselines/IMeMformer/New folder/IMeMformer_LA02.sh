#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --job-name=HaiyangJiang
#SBATCH --time=168:00:00
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:h100:1
#SBATCH --account=a_hongzhi_yin
#SBATCH -o outputfile/IMeMfLA02.output
#SBATCH -e outputfile/IMeMfLA02.error
#SBATCH -J IMeMfLA02

module load miniconda3/4.12.0
source $EBROOTMINICONDA3/etc/profile.d/conda.sh
source activate /home/s4841505/.conda/envs/basicts/

srun python experiments/train.py -c baselines/IMeMformer/METR-LA02.py --gpus '0'
