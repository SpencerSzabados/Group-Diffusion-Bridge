# User launch paramters
DATASET=$1
NGPU=$2
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
USE_16FP=False
LOG_INTERVAL=500
SAMPLE_INTERVAL=20000
SAVE_INTERVAL=20000
TOTAL_TRAINING_STEPS=200000
AUG=0


# Arguments
if [[ $DATASET == "NAME" ]]; then
    DATA_DIR=YOUR_DATASET_PATH
    IMAGE_SIZE=64
    SAVE_INTERVAL=100000
elif [[ $DATASET == "fives" ]]; then
    WORK_DIR=/u6/sszabado/checkpoints/ddbm/vae/
    DATA_DIR=/u6/sszabado/datasets/fives512_patches/train/images/,/u6/sszabado/datasets/fives512_patches/train/masks/
    IMAGE_SIZE=512
    USE_16FP=False
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