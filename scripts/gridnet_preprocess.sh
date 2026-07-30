#!/bin/bash
#SBATCH --job-name test-pre # Name for your job
#SBATCH --nodes 1             # do not change (unless you do MPI)
#SBATCH --nodelist lts2srv4 
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 6      # reserve 4 cpus on a single node
#SBATCH --time 1000             # Runtime in minutes.
#SBATCH --mem 40000             # Reserve 10 GB RAM for the job
#SBATCH --partition gpu         # Partition to submit ('gpu' or 'cpu')
#SBATCH --qos students-msc             # QOS ('staff' or 'students')
#SBATCH --output preprocessing-%j.txt       # Standard out goes to this file
#SBATCH --error preprocessing-%j.txt        # Standard err goes to this file
#SBATCH --mail-user guillaume.spahr@epfl.ch     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc
#SBATCH --gres gpu:titanrtx:1            # Reserve 1 GPU for usage, can be 'teslak40', 'gtx1080', or 'titanrtx'
#SBATCH --chdir /nfs_home/gspahr/PDM

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate ptv3

ROOT_DIR="/mnt/scratch/lts2/aspert/datasets/lidar/GridNet"
# SPLIT_JSON="$ROOT_DIR/split.json"
SPLIT_JSON="/nfs_home/gspahr/PDM/split_.json"
OUTPUT_DIR="/mnt/scratch/students/gspahr/GridNet_preprocessed"


# for SPLIT in train val; do
for SPLIT in test; do
    # if [ "$SPLIT" = "test" ]; then
    #     VOXEL_SIZE=0.0
    #     MIN_PTS=1
    # else
    #     VOXEL_SIZE=0.02
    #     MIN_PTS=10000
    # fi
    if [ "$SPLIT" = "test" ]; then
        VOXEL_SIZE=0.0
        CHUNK_SIZE=18
        STRIDE=10
        MIN_PTS=100
    elif [ "$SPLIT" = "val" ]; then
        VOXEL_SIZE=0.01
        CHUNK_SIZE=18
        STRIDE=16
        MIN_PTS=1000
    else
        VOXEL_SIZE=0.03
        CHUNK_SIZE=18
        STRIDE=10
        MIN_PTS=10000
    fi
    python -m datasets.preprocessing.gridnet.gridnet_preprocess \
        --raw-root   "$ROOT_DIR"        \
        --out-root   "$OUTPUT_DIR"      \
        --split-json "$SPLIT_JSON"      \
        --split      "$SPLIT"           \
        --target-h    768               \
        --target-w    1008              \
        --chunk-size  "$CHUNK_SIZE"     \
        --stride      "$STRIDE"         \
        --min-pts     "$MIN_PTS"        \
        --max-cam     10                \
        --min-visible 500               \
        --n-candidates 60               \
        --depth-buffer 3                \
        --depth-thresh 0.75             \
        --voxel-size   "$VOXEL_SIZE"    \
        --proj-batch-size 65536         \
        --workers      3           
done