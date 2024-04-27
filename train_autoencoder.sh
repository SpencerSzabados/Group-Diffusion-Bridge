DATASET=$1
NGPU=$2

source ./args_autoencoder.sh $DATASET $NGPU

# CUDA_VISIBLE_DEVICES=0 python train_autoencoder.py \
#     --workdir $WORK_DIR --data_dir $DATA_DIR --image_size $IMAGE_SIZE --global_batch_size $BATCH_SIZE \
#     --lr $LR --ema_rate $EMA_RATE --weight_decay $WEIGHT_DECAY --lr_anneal_steps $LR_ANNEAL_STEPS \
#     --log_interval $LOG_INTERVAL --save_interval $SAVE_INTERVAL --total_training_steps $TOTAL_TRAINING_STEPS \
#     --augment $AUG --ngpu $NGPU

mpiexec --use-hwthread-cpus --oversubscribe -n $NGPU python train_autoencoder.py \
    --workdir $WORK_DIR --data_dir $DATA_DIR --image_size $IMAGE_SIZE --global_batch_size $BATCH_SIZE \
    --lr $LR --ema_rate $EMA_RATE --weight_decay $WEIGHT_DECAY --lr_anneal_steps $LR_ANNEAL_STEPS \
    --log_interval $LOG_INTERVAL --save_interval $SAVE_INTERVAL --total_training_steps $TOTAL_TRAINING_STEPS \
    --augment $AUG --ngpu 1 -fp16 $USE_16FP