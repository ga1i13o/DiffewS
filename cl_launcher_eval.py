import os
from os.path import join
import argparse
import sys


def main(id_str, benchmark='coco'):
    folder = "/leonardo_work/IscrB_LarGEO/gtrivigno/DiffewS"
    if not os.path.abspath(os.curdir) == folder: sys.exit()
    all_exps = os.listdir('logs_v3')

    all_exps = list(filter(lambda x: id_str in x and 'eval' not in x, all_exps))
    for ie, exp in enumerate(all_exps):
        exp_name = exp
        fold = exp_name.split('fold')[-1].split('.')[0]
        exp_name = f"eval_{exp_name.replace('.', '_')}"
        r_path = join('logs_v3', exp, 'checkpoint-20000', 'unet')

        CONTENT = \
f"""#!/bin/bash 
#SBATCH -A IscrB_LarGEO
#SBATCH --job-name=EXP_NAME
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --output {folder}/out_files/out_EXP_NAME.txt
#SBATCH --error {folder}/out_files/err_EXP_NAME.txt
module load python/3.10.8--gcc--11.3.0
export WANDB_MODE=offline
source /leonardo/home/userexternal/gtrivign/diffews/bin/activate
cd {folder}
python evaluation_util/main_oss.py \
 --log-root logs/EXP_NAME \
 --denoise_steps 1 \
 --checkpoint /leonardo/home/userexternal/gtrivign/.cache/modelscope/hub/models/zzzmmz/Diffews/weight/stable-diffusion-2-1-ref8inchannels-tag4inchannels \
 --unet_ckpt_path {r_path} \
 --datapath /leonardo_scratch/fast/IscrB_LarGEO/fast_dataset \
 --benchmark {benchmark} \
 --img-size 512 \
  --ensemble_size 1 \
 --bsz 1 \
 --scheduler_load_path ./scheduler_1.0_1.0 \
 --nshot 1 \
 --fold {fold} \
 --threshold 0 \
 --r_threshold 0.25 

"""

        filename = f"{folder}/jobs/{exp_name}.sh"
        content = CONTENT.replace("EXP_NAME", exp_name)
        with open(filename, "w") as file:
            _ = file.write(content)
        _ = os.system(f"sbatch {filename}")
        print(f"sbatch {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('id_str', type=str, help='')
    parser.add_argument('--ds', type=str, default=None, help='')

    args = parser.parse_args()
    main(args.id_str, args.ds)
    