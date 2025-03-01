export PYTHONPATH=./
TIME=$(date "+%Y%m%d_%H_%M_%S")

THRESHOLD=0.1
NSHOT=1
FOLD=0
ITER=14000

export MODEL_DIR=$1
# export MODEL_NAME=$(basename "$MODEL_DIR")
export MODEL_NAME="unet"
echo $MODEL_NAME
# FOLD=$2
for THRESHOLD in 0.25; do
OUTPUT_DIR="./logs/icl_seg_eval_${ITER}/${MODEL_NAME}/rthreshold${THRESHOLD}_${NSHOT}shot_fold${FOLD}_iter${ITER}_concat-kv_ns"
python evaluation_util/main_oss.py \
 --log-root $OUTPUT_DIR \
 --denoise_steps 1 \
 --checkpoint /home/gtrivigno/.cache/modelscope/hub/models/zzzmmz/Diffews/weight/stable-diffusion-2-1-ref8inchannels-tag4inchannels \
 --unet_ckpt_path $MODEL_DIR/unet \
 --datapath /data/datasets \
 --benchmark 'pascal_cd' \
 --img-size 512 \
  --ensemble_size 1 \
 --bsz 1 \
 --scheduler_load_path ./scheduler_1.0_1.0 \
 --nshot $NSHOT \
 --fold $FOLD \
 --threshold 0 \
 --r_threshold $THRESHOLD 
done


# CUDA_VISIBLE_DEVICES=5 bash  scripts/eval_coco2014_rthres_1shot_nosample.sh weight/coco_fold0
# CUDA_VISIBLE_DEVICES=2 bash  scripts/eval_coco2014_rthres_5shot_nosample.sh weight/coco_fold1
# CUDA_VISIBLE_DEVICES=3 bash  scripts/eval_coco2014_rthres_5shot_nosample.sh weight/coco_fold2
# CUDA_VISIBLE_DEVICES=4 bash  scripts/eval_coco2014_rthres_5shot_nosample.sh weight/coco_fold3
