# res768 bs64

res=512
lr=1e-5
#lr=1e-4
wr=0
bs=1
gas=4 # bs = 8 * bs * gas
step=20000
task=v3
fold=3
shot=1
output_dir=./logs_v3/icl_coco_fold${fold}_res${res}_lr${lr}_wr${wr}_step${step}_bs${bs}_gas${gas}_gpu2_nearest_seed${seed}_shot${shot}

echo ${output_dir}

for fold in 0 ; do
CUDA_VISIBLE_DEVICES='1'  accelerate launch --num_processes 1 --main_process_port 1234 --mixed_precision "fp16" --num_machines 1 \
train_tools/train_icl_multitask_nocrop_nearest_nshot_v3.py \
 --mixed_precision="fp16" \
 --train_batch_size=${bs} \
 --benchmark='pascal_part' \
 --checkpointing_steps 2000 \
 --pretrained_model_name_or_path="/home/gtrivigno/.cache/modelscope/hub/models/zzzmmz/Diffews/weight/stable-diffusion-2-1-ref8inchannels-tag4inchannels" \
 --output_dir=${output_dir} \
 --train_data_dir "/data/datasets" \
 --resolution=${res} \
 --learning_rate=${lr} \
 --lr_warmup_steps ${wr} \
 --max_train_steps=${step} \
 --validation_steps 2000 \
 --lr_scheduler polynomial \
 --lr_scheduler_power 1.0 \
 --gradient_accumulation_steps=${gas} \
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
 --dataloader_num_workers=16 \
 --checkpoints_total_limit 10 \
 --nshot ${shot} \
 --fold=${fold} \
 --scheduler_load_path ./scheduler_1.0_1.0 
done
