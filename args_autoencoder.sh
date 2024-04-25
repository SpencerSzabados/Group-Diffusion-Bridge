# User launch paramters
DATASET=$1
# Default values
WORK_DIR=""
DATASET_DIR=""
IMAGE_SIZE=64
NUM_IMG_CH=3
BATCH_SIZE=64
LR=1e-4
EMA_RATE=0.9999
WEIGHT_DECAY=0.0
LR_ANNEAL_STEPS=0
LOG_INTERVAL=50
SAVE_INTERVAL=100
TOTAL_TRAINING_STEPS=100000
AUG=0
NGPU=2


# Arguments
if [[ $DATASET == "NAME" ]]; then
    DATA_DIR=YOUR_DATASET_PATH
    IMAGE_SIZE=64
    SAVE_ITER=100000
elif [[ $DATASET == "fives" ]]; then
    WORK_DIR=/home/checkpoints/group-diffusion-bridge/temp/
    DATA_DIR=/home/datasets/fives512_patches/train/
    IMAGE_SIZE=512
else
    echo "Not supported"
    exit 1    
fi


if [[ $IMAGE_SIZE == 512 ]]; then
    BATCH_SIZE=2
elif  [[ $IMAGE_SIZE == 256 ]]; then
    BATCH_SIZE=16
elif  [[ $IMAGE_SIZE == 128 ]]; then
    BATCH_SIZE=14
elif  [[ $IMAGE_SIZE == 64 ]]; then
    BATCH_SIZE=30
else
    echo "Not supported"
    exit 1
fi