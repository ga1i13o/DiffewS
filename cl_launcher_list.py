
import os
import sys

folder = "/leonardo_work/IscrB_LarGEO/gtrivigno/DiffewS"
if not os.path.abspath(os.curdir) == folder: sys.exit()

NGPUS = 4
BS = 1
grid = [    
    # {'name': f'diffews_fss_fold0', ' --benchmark': 'fss'},    

    # {'name': f'diffews_pascal_part_fold0', ' --benchmark': 'pascal_part'},    
    # {'name': f'diffews_pascal_part_fold1', ' --benchmark': 'pascal_part'},    
    # {'name': f'diffews_pascal_part_fold2', ' --benchmark': 'pascal_part'},    
    # {'name': f'diffews_pascal_part_fold3', ' --benchmark': 'pascal_part'},    

    # {'name': f'diffews_paco_part_fold0', ' --benchmark': 'paco_part'},    
    # {'name': f'diffews_paco_part_fold1', ' --benchmark': 'paco_part'},    
    # {'name': f'diffews_paco_part_fold2', ' --benchmark': 'paco_part'},    
    # {'name': f'diffews_paco_part_fold3', ' --benchmark': 'paco_part'},    

    # {'name': f'diffews_lvis_fold0', ' --benchmark': 'lvis'},    
    # {'name': f'diffews_lvis_fold1', ' --benchmark': 'lvis'},    
    # {'name': f'diffews_lvis_fold2', ' --benchmark': 'lvis'},    
    # {'name': f'diffews_lvis_fold3', ' --benchmark': 'lvis'},    
    # {'name': f'diffews_lvis_fold4', ' --benchmark': 'lvis'},    
    # {'name': f'diffews_lvis_fold5', ' --benchmark': 'lvis'},    
    # {'name': f'diffews_lvis_fold6', ' --benchmark': 'lvis'},    
    # {'name': f'diffews_lvis_fold7', ' --benchmark': 'lvis'},    
    # {'name': f'diffews_lvis_fold8', ' --benchmark': 'lvis'},    
    # {'name': f'diffews_lvis_fold9', ' --benchmark': 'lvis'},    

    {'name': f'diffews_pascal_fold0', ' --benchmark': 'pascal'},    
    {'name': f'diffews_pascal_fold1', ' --benchmark': 'pascal'},    
    {'name': f'diffews_pascal_fold2', ' --benchmark': 'pascal'},    
    {'name': f'diffews_pascal_fold3', ' --benchmark': 'pascal'},    
]

for arg_set in grid:
    add_args = ''
    exp_name = arg_set['name']
    for arg_s, arg_v in arg_set.items():
        if arg_s != 'name':
            add_args += f'{arg_s} {arg_v}'
        
    fold = int(exp_name.split('_fold')[-1].split('_')[0])
    add_args += f' --fold {fold}'
    CONTENT = \
f"""#!/bin/bash 
#SBATCH -A IscrC_AstroGeo
#SBATCH --job-name=EXP_NAME
#SBATCH --gres=gpu:{NGPUS}
#SBATCH -N 1
#SBATCH --ntasks-per-node={NGPUS*8}
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=100GB
#SBATCH --time=24:00:00
#SBATCH --output {folder}/out_files/out_EXP_NAME.txt
#SBATCH --error {folder}/out_files/err_EXP_NAME.txt
module load python/3.10.8--gcc--11.3.0
export WANDB_MODE=offline
source /leonardo/home/userexternal/gtrivign/diffews/bin/activate
cd {folder}
port=$(python get_free_port.py)
accelerate launch --num_processes {NGPUS} --main_process_port ${{port}} --mixed_precision "fp16" --num_machines 1 \
    train_tools/train_icl_multitask_nocrop_nearest_nshot_v3.py \
    --mixed_precision="fp16" \
    --train_batch_size={BS} \
    --checkpointing_steps 2000 \
    --pretrained_model_name_or_path="/leonardo/home/userexternal/gtrivign/.cache/modelscope/hub/models/zzzmmz/Diffews/weight/stable-diffusion-2-1-ref8inchannels-tag4inchannels" \
    --output_dir=logs_v3/EXP_NAME \
    --train_data_dir "/leonardo_scratch/fast/IscrB_LarGEO/fast_dataset" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --lr_warmup_steps=0 \
    --max_train_steps=20000 \
    --validation_steps 2000 \
    --lr_scheduler polynomial \
    --lr_scheduler_power 1.0 \
    --gradient_accumulation_steps=2 \
    --enable_xformers_memory_efficient_attention \
    --max_grad_norm=1.0 \
    --adam_weight_decay=1e-2 \
    --tracker_project_name sd21_train_dis \
    --seed=42 \
    --image_ref_column img_ref \
    --image_tag_column img_tag \
    --conditioning_image_ref_column ref_conditioning_image \
    --conditioning_image_tag_column tag_conditioning_image \
    --caption_column 'text' \
    --cache_dir './cache' \
    --allow_tf32 \
    --dataloader_num_workers=8 \
    --checkpoints_total_limit 8 \
    --nshot 1 \
    --scheduler_load_path ./scheduler_1.0_1.0 {add_args}
"""

    filename = f"{folder}/jobs/{exp_name}.sh"
    content = CONTENT.replace("EXP_NAME", exp_name)
    with open(filename, "w") as file:
        _ = file.write(content)
    _ = os.system(f"sbatch {filename}")
    print(f"sbatch {filename}")


