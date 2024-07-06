echo "Job stared..."

source ./args.sh $DATASET_NAME fives $PRED vp $NGPU 1

NCCL_P2P_LEVEL=NVL mpiexec --use-hwthread-cpus --oversubscribe -n $NGPU python eval_ddbm.py \
    --work_dir=$WORK_DIR --exp=$EXP --data_dir=$DATA_DIR --dataset=$DATASET \
    --image_size $EMB_SIZE --in_channels $EMB_CHANNELS --data_image_size $DATA_IMG_SIZE --data_image_channels $DATA_IMG_CHANNELS \
    --attention_resolutions $ATTN --class_cond False --use_scale_shift_norm True \
    --lr 0.0001 --dropout 0.1 --ema_rate 0.9999 --batch_size $BS \
    --num_channels $NUM_CH --num_head_channels $NUM_HEADS \
    --num_res_blocks $NUM_RES_BLOCKS --resblock_updown True \
    ${COND:+ --condition_mode="${COND}"} ${MICRO:+ --microbatch="${MICRO}"} \
    --pred_mode=$PRED  --schedule_sampler $SAMPLER ${UNET:+ --unet_type="${UNET}"} \
    --use_fp16 $USE_16FP --attention_type $ATTN_TYPE --weight_decay 0.0 \
    --weight_schedule bridge_karras \
    ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"}  \
    ${CH_MULT:+ --channel_mult="${CH_MULT}"} \
    --num_workers=$NGPU --sigma_data $SIGMA_DATA --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN --cov_xy $COV_XY \
    --log_interval=$LOG_INTERVAL --test_interval=$TEST_INTERVAL --save_interval=$SAVE_INTERVAL \
    --debug False \
    ${CKPT:+ --resume_checkpoint="${CKPT}"} \
    --dice_weight=$DICE_WEIGHT --dice_tol=$DICE_TOL &