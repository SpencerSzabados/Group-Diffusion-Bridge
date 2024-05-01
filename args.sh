# Default values
BS=64
DATASET_NAME=$1
PRED=$2
NGPU=$3
SIGMA_MAX=80.0
SIGMA_MIN=0.002
SIGMA_DATA=0.5
COV_XY=0
NUM_CH=256
NUM_HEADS=64
ATTN=32,16,8
SAMPLER=real-uniform
NUM_RES_BLOCKS=2
USE_16FP=True
ATTN_TYPE=flash
TEST_INTERVAL=10000
IN_CHANNELS=3
OUT_CHANNELS=3


# Arguments
if [[ $DATASET_NAME == 'vae_fives_patches' ]]; then
    DATA_DIR=/home/datasets/fives512_patches/
    WORK_DIR=/home/checkpoints/ddbm/
    DATASET=fives_patches
    DATA_IMG_SIZE=512
    DATA_IMG_CHANNELS=3
    OUT_CHANNELS=1
    EMB_SIZE=64
    EMB_CHANNELS=4
    NUM_CH=192
    NUM_RES_BLOCKS=3
    EXP="h2e_rot90_${DATASET}_${IMG_SIZE}_${IN_CHANNELS}ich_${NUM_CH}ch_${NUM_RES_BLOCKS}b"
    SAVE_ITER=10000
fi
    

if  [[ $PRED == "ve" ]]; then
    EXP+="_ve"
    COND=concat
elif  [[ $PRED == "vp" ]]; then
    EXP+="_vp"
    COND=concat
    BETA_D=2
    BETA_MIN=0.1
    SIGMA_MAX=1
    SIGMA_MIN=0.0001
elif  [[ $PRED == "ve_simple" ]]; then
    EXP+="_ve_simple"
    COND=concat
elif  [[ $PRED == "vp_simple" ]]; then
    EXP+="_vp_simple"
    COND=concat
    BETA_D=2
    BETA_MIN=0.1
    SIGMA_MAX=1
    SIGMA_MIN=0.0001
else
    echo "Not supported"
    exit 1
fi


if  [[ $DATA_IMG_SIZE == 512 ]]; then
    BS=8
elif  [[ $DATA_IMG_SIZE == 256 ]]; then
    BS=16
elif  [[ $DATA_IMG_SIZE == 128 ]]; then
    BS=14
elif  [[ $DATA_IMG_SIZE == 64 ]]; then
    BS=30
else
    echo "Not supported"
    exit 1
fi