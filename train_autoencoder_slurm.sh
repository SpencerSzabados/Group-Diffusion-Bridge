#!/bin/bash
#SBATCH --job-name=fine_tune_sd_vae
#SBATCH --nodes=1
#SBATCH --gres=gpu:yaolianggpu:2 -p YAOLIANG
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --signal=B:USR1@30
#SBATCH --time=06:00:00
#SBATCH --output=%x.out
#SBATCH --error=%x.err

# Launch Python script in background
echo "Job stared..."
bash train_autoencoder.sh $DATASET fives $NGPU 2 &

# Capture the PID of the Python process
PID=$!

# Define the cleanup function to handle termination signals
cleanup() {
    # Send signal USR1 to Python script with a delay of 180 seconds
    echo "Received termination signal, handling it gracefully..."
    kill -USR1 $PID
}

# Trap termination signals and call the cleanup function
trap cleanup USR1

# Wait for Python process to finish
wait $PID
