#!/bin/bash
#SBATCH --job-name=fine_tune_sd_vae
#SBATCH --nodes=1
#SBATCH --gres=gpu:yaolianggpu:1 -p YAOLIANG
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --signal=B:SIGUSR1@30
#SBATCH --time=12:00:00
#SBATCH --output=%x.out
#SBATCH --error=%x.err

# Launch Python script in background
echo "Job stared..."

source activate ddbm

source ./args_autoencoder.sh $DATSET fives $NGPU 1

NCCL_P2P_LEVEL=NVL mpiexec --use-hwthread-cpus --oversubscribe -n $NGPU python train_autoencoder.py \
    --workdir $WORK_DIR --data_dir $DATA_DIR --image_size $IMAGE_SIZE --global_batch_size $BATCH_SIZE \
    --lr $LR --ema_rate $EMA_RATE --weight_decay $WEIGHT_DECAY --lr_anneal_steps $LR_ANNEAL_STEPS \
    --log_interval $LOG_INTERVAL --save_interval $SAVE_INTERVAL --total_training_steps $TOTAL_TRAINING_STEPS \
    --augment $AUG --num_workers 1 --fp16 $USE_16FP &

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
