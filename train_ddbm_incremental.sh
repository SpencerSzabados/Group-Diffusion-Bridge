DATASET_NAME=$1
PRED=$2
CKPT=$3


source ./args.sh $DATASET_NAME $PRED

FREQ_SAVE_ITER=20000
NGPU=1

# mpiexec --allow-run-as-root -n $NGPU python train_ddbm_incremental.py --exp=$EXP \
#     --attention_resolutions $ATTN --class_cond False --use_scale_shift_norm True \
#     --lr 0.0001 --dropout 0.1 --ema_rate 0.9999 --batch_size $BS \
#     --image_size $IMG_SIZE --in_channels $IN_CHANNELS --num_channels $NUM_CH --num_head_channels $NUM_HEADS \
#     --num_res_blocks $NUM_RES_BLOCKS --resblock_updown True \
#     ${COND:+ --condition_mode="${COND}"} ${MICRO:+ --microbatch="${MICRO}"} \
#     --pred_mode=$PRED  --schedule_sampler $SAMPLER ${UNET:+ --unet_type="${UNET}"} \
#     --use_fp16 $USE_16FP --attention_type $ATTN_TYPE --weight_decay 0.0 \
#     --weight_schedule bridge_karras\
#     ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"}  \
#     --data_dir=$DATA_DIR --dataset=$DATASET ${CH_MULT:+ --channel_mult="${CH_MULT}"} \
#     --num_workers=8  --sigma_data $SIGMA_DATA --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN --cov_xy $COV_XY \
#     --test_interval=$TEST_INTERVAL --save_interval_for_preemption=$FREQ_SAVE_ITER --save_interval=$SAVE_ITER \
#     --debug=False \
#     ${CKPT:+ --resume_checkpoint="${CKPT}"} \
#     --dice_weight 0.5 --dice_tol 0.5

CUDA_VISIBLE_DEVICES=0 python train_ddbm_incremental.py --exp=$EXP \
    --attention_resolutions $ATTN --class_cond False --use_scale_shift_norm True \
    --lr 0.0001 --dropout 0.1 --ema_rate 0.9999 --batch_size $BS \
    --image_size $IMG_SIZE --in_channels $IN_CHANNELS --num_channels $NUM_CH --num_head_channels $NUM_HEADS \
    --num_res_blocks $NUM_RES_BLOCKS --resblock_updown True \
    ${COND:+ --condition_mode="${COND}"} ${MICRO:+ --microbatch="${MICRO}"} \
    --pred_mode=$PRED  --schedule_sampler $SAMPLER ${UNET:+ --unet_type="${UNET}"} \
    --use_fp16 $USE_16FP --attention_type $ATTN_TYPE --weight_decay 0.0 \
    --weight_schedule bridge_karras\
    ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"}  \
    --data_dir=$DATA_DIR --dataset=$DATASET ${CH_MULT:+ --channel_mult="${CH_MULT}"} \
    --num_workers=8  --sigma_data $SIGMA_DATA --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN --cov_xy $COV_XY \
    --test_interval=$TEST_INTERVAL --save_interval_for_preemption=$FREQ_SAVE_ITER --save_interval=$SAVE_ITER \
    --debug=False \
    ${CKPT:+ --resume_checkpoint="${CKPT}"} \
    --dice_weight 0.5 --dice_tol 0.5