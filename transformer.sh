#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --account=lonepeak-gpu
#SBATCH --partition=lonepeak-gpu
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --mem=15GB
#SBATCH -e sigma0.5transformer_train_slurm-%j.err-%N
#SBATCH -o sigma0.5transformer_train_slurm-%j.out-%N

source activate base
conda activate cifar10


#python /uufs/chpc.utah.edu/common/home/u1110463/RF_projects/rf_transformer/transformer_train.py --data_root /uufs/chpc.utah.edu/common/home/u1110463/RF_projects/rf_transformer/Datasets --logging_root /uufs/chpc.utah.edu/common/home/u1110463/RF_projects/rf_transformer/log_root --train_test train --sigma 0.3 2>&1 >> /uufs/chpc.utah.edu/common/home/u1110463/RF_projects/rf_transformer/log_root/sigma0.5_optimized_transformer1.log
python /uufs/chpc.utah.edu/common/home/u1110463/RF_projects/rf_transformer/transformer_test.py --data_root './Datasets'  --checkpoint './log_root/logs/02_22/05-59-27_data_root_Datasets_logging_root_log_root_experiment_name__checkpoint_None_sigma_0.3_lr_0.001_reg_weight_0.0_/model-epoch_2_iter_9000.pth' --train_test test>> /uufs/chpc.utah.edu/common/home/u1110463/RF_projects/rf_transformer/log_root/hardware_test8_transformer.log
