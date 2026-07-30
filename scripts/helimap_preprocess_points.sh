#!/bin/bash
#SBATCH --job-name points # Name for your job
#SBATCH --nodes 1             # do not change (unless you do MPI)
#SBATCH --nodelist lts2srv5 
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8      # reserve 4 cpus on a single node
#SBATCH --time 2880             # Runtime in minutes.
#SBATCH --mem 40000             # Reserve 50 GB RAM for the job
#SBATCH --partition cpu         # Partition to submit ('gpu' or 'cpu')
#SBATCH --qos students-msc             # QOS ('staff' or 'students')
#SBATCH --output heli-points-%j.txt       # Standard out goes to this file
#SBATCH --error heli-points-err-%j.txt        # Standard err goes to this file
#SBATCH --mail-user guillaume.spahr@epfl.ch     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc
#SBATCH --chdir /nfs_home/gspahr/PDM

# ACTIVATE ANACONDA
if ! command -v conda >/dev/null 2>&1; then
  for _c in "$CONDA_BASE" /opt/conda "$HOME/miniconda3" "$HOME/anaconda3" /nfs_home/gspahr/miniconda3; do
    [ -n "$_c" ] && [ -f "$_c/etc/profile.d/conda.sh" ] && source "$_c/etc/profile.d/conda.sh" && break
  done
fi
eval "$(conda shell.bash hook)"
conda activate ptv3
# Fail loudly instead of silently dropping to system Python 2.7.
python -c 'import sys; assert sys.version_info[0] >= 3, "conda not active: " + sys.version' || exit 1

##################
#    Atocha Dataset
##################
ROOT_DIR="/net/lts2srv4/mnt/scratch/students/gspahr/lidar/ATOCHA"
LAS_DIR="$ROOT_DIR/LIDAR/LAS14"
OUTPUT_DIR="/net/lts2srv4//mnt/scratch/students/gspahr/ATOCHA"
SPLIT_JSON="$OUTPUT_DIR/split.json"

CAMERA_POS="$ROOT_DIR/IMG/CAMERA_POS_ATOC.xml"
CAMERA_FRONT="$ROOT_DIR/IMG/IXM100-50mmFRONT_ATOC.xml"
CAMERA_FRONT_FOLDER="$ROOT_DIR/IMG/03_FRONT"
CAMERA_NADIR="$ROOT_DIR/IMG/IXMGS120-35mm_NADIR_ATOC.xml"
CAMERA_NADIR_FOLDER="$ROOT_DIR/IMG/01_NADIR"

# python -m datasets.preprocessing.helimap.make_split_json \
#     --dataset        atocha        \
#     --las-dir        "$LAS_DIR"    \
#     --out            "$SPLIT_JSON"  \
#     --train-ratio    0.6            \
#     --val-ratio      0.2           \
#     --test-ratio     0.2           \
#     --ignore-names   unclassified


# # Re-run point pass for val/test so chunks gain orig_idx.npy + n_las_pts for
# # prediction reconstruction.  Train doesn't need it (no eval reconstruction).
# for SPLIT in train val test; do
#     if [ "$SPLIT" = "test" ]; then
#         VOXEL_SIZE=0.0
#         CHUNK_SIZE=18
#         STRIDE=12
#         MIN_PTS=100
#     elif [ "$SPLIT" = "val" ]; then
#         VOXEL_SIZE=0.01
#         CHUNK_SIZE=18
#         STRIDE=11
#         MIN_PTS=1000
#     else
#         VOXEL_SIZE=0.03
#         CHUNK_SIZE=18
#         STRIDE=9
#         MIN_PTS=10000
#     fi
#     python -m datasets.preprocessing.helimap.helimap_preprocess_points \
#         --dataset    atocha         \
#         --las-dir    "$LAS_DIR"     \
#         --split-json "$SPLIT_JSON"  \
#         --split      "$SPLIT"       \
#         --out-root   "$OUTPUT_DIR"  \
#         --chunk-size  "$CHUNK_SIZE" \
#         --stride      "$STRIDE"     \
#         --min-pts     "$MIN_PTS"    \
#         --voxel-size  "$VOXEL_SIZE" \
#         --compute-normals
# done

##################
#    A9CR Dataset
##################
ROOT_DIR="/net/lts2srv4/mnt/scratch/students/gspahr/lidar/A9CR"
LAS_DIR="$ROOT_DIR/LIDAR"
OUTPUT_DIR="/net/lts2srv4/mnt/scratch/students/gspahr/A9CR"
SPLIT_JSON="$OUTPUT_DIR/split.json"

CAMERA_POS="$ROOT_DIR/IMG/CAMERA_POS_A9CR.xml"
CAMERA_NADIR="$ROOT_DIR/IMG/IXMRS150-40mm_A9CR_NADIR.xml"
CAMERA_NADIR_FOLDER="$ROOT_DIR/IMG/01_NADIR"

python -m datasets.preprocessing.helimap.make_split_json \
    --dataset        a9cr          \
    --las-dir        "$LAS_DIR"    \
    --out            "$SPLIT_JSON"  \
    --train-ratio    0.6            \
    --val-ratio      0.20           \
    --test-ratio     0.20           \
    --ignore-names   unclassified

for SPLIT in train val test; do
    if [ "$SPLIT" = "test" ]; then
        VOXEL_SIZE=0.02
        CHUNK_SIZE=14
        STRIDE=10
        MIN_PTS=100
        MIN_LABELED_FRAC=0.0      # eval must stay representative / cover all points
    elif [ "$SPLIT" = "val" ]; then
        VOXEL_SIZE=0.03
        CHUNK_SIZE=14
        STRIDE=10
        MIN_PTS=1000
        MIN_LABELED_FRAC=0.01      # eval must stay representative
    else
        VOXEL_SIZE=0.03
        CHUNK_SIZE=14
        STRIDE=10
        MIN_PTS=10000
        MIN_LABELED_FRAC=0.01     # train-only: drop label-sparse chunks (<1% labelled)
    fi
    python -m datasets.preprocessing.helimap.helimap_preprocess_points \
        --dataset    a9cr          \
        --las-dir    "$LAS_DIR"   \
        --split-json "$SPLIT_JSON" \
        --split      "$SPLIT"      \
        --out-root   "$OUTPUT_DIR" \
        --chunk-size  "$CHUNK_SIZE" \
        --stride      "$STRIDE"     \
        --min-pts     "$MIN_PTS"    \
        --voxel-size  "$VOXEL_SIZE" \
        --min-labeled-frac "$MIN_LABELED_FRAC" 
        # --compute-normals
done