#!/bin/sh
#SBATCH --job-name=net_training            # Название задачи
##SBATCH --error=my_numpy_task-%j.err        # Файл для вывода ошибок
##SBATCH --output=my_numpy_task-%j.log       # Файл для вывода результатов
#SBATCH -o my.stdout
#SBATCH --time=1:00:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=8                # Выполнение расчёта на 2 ядрах CPU
#SBATCH --gpus=1                            # Требуемое кол-во GPU
module load slurm
python src/main.py --dataset_path=../../datasets/CAMUS --config=configs/ventricle.json --use_wandb=True --wandb_run_name=slrm
