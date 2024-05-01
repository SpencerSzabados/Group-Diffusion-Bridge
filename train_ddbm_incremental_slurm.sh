#!/bin/bash
#SBATCH --job-name=ddbm_vae
#SBATCH --nodes=1
#SBATCH --gres=gpu:yaolianggpu:2 -p YAOLIANG
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --signal=B:USR1@30
#SBATCH --time=12:00:00
#SBATCH --output=%x.out
#SBATCH --error=%x.err

# Launch Python script in background
echo "Job stared..."

source ./args.sh $DATSET vae_fives_patches $PRED vp $NGPU 2

mpiexec --use-hwthread-cpus --oversubscribe -n $NGPU python train_ddbm_incremental.py \
    --work_dir=$WORK_DIR --exp=$EXP --data_dir=$DATA_DIR --dataset=$DATASET \
    --image_size $EMB_SIZE --in_channels $EMB_CHANNELS --data_image_size $DATA_IMG_SIZE --data_image_channels $DATA_IMG_CHANNELS \
    --attention_resolutions $ATTN --class_cond False --use_scale_shift_norm True \
    --lr 0.0001 --dropout 0.1 --ema_rate 0.9999 --batch_size $BS \
    --num_channels $NUM_CH --num_head_channels $NUM_HEADS \
    --num_res_blocks $NUM_RES_BLOCKS --resblock_updown True \
    ${COND:+ --condition_mode="${COND}"} ${MICRO:+ --microbatch="${MICRO}"} \
    --pred_mode=$PRED  --schedule_sampler $SAMPLER ${UNET:+ --unet_type="${UNET}"} \
    --use_fp16 $USE_16FP --attention_type $ATTN_TYPE --weight_decay 0.0 \
    --weight_schedule bridge_karras\
    ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"}  \
    ${CH_MULT:+ --channel_mult="${CH_MULT}"} \
    --num_workers=8 --sigma_data $SIGMA_DATA --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN --cov_xy $COV_XY \
    --test_interval=$TEST_INTERVAL --save_interval=$SAVE_ITER \
    --debug=False \
    ${CKPT:+ --resume_checkpoint="${CKPT}"} \
    --dice_weight 0.5 --dice_tol 0.5

# Capture the PID of the Python process
PID=$!
echo "Captured PID: $PID"

# Define the cleanup function to handle termination signals
cleanup() {
    # Send signal USR1 to Python script with a delay of 180 seconds
    echo "Received termination signal, handling it gracefully..."
    kill -SIGUSR1 $PID
}

# Trap termination signals and call the cleanup function
trap cleanup SIGUSR1

# Wait for Python process to finish
wait $PID
