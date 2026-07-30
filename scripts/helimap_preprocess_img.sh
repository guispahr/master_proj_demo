#!/bin/bash
#SBATCH --job-name heli-img # Name for your job
#SBATCH --nodes 1             # do not change (unless you do MPI)
#SBATCH --nodelist lts2srv5 
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8      # reserve 4 cpus on a single node
#SBATCH --time 2880             # Runtime in minutes.
#SBATCH --mem 40000             # Reserve 50 GB RAM for the job
#SBATCH --partition cpu         # Partition to submit ('gpu' or 'cpu')
#SBATCH --qos students-msc             # QOS ('staff' or 'students')
#SBATCH --output heli-img-%j.txt       # Standard out goes to this file
#SBATCH --error heli-img-err-%j.txt        # Standard err goes to this file
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

python -m datasets.preprocessing.helimap.helimap_preprocess_images \
    --out-root          "$OUTPUT_DIR"                       \
    --splits            train val test                      \
    --poses             "$CAMERA_POS"                       \
    --cal-front         "$CAMERA_FRONT"                     \
    --images-front      "$CAMERA_FRONT_FOLDER"              \
    --max-cam-front     8                                   \
    --cal-nadir         "$CAMERA_NADIR"                     \
    --images-nadir      "$CAMERA_NADIR_FOLDER"              \
    --max-cam-nadir     6                                   \
    --target-h          1040                                 \
    --target-w          1040                                 \
    --margin-scale      1.2                                 \
    --min-crop-px       500                                 \
    --min-visible       1000                                 \
    --n-candidates      40                                  \
    --depth-buffer      3                                   \
    --depth-thresh      0.75                                \
    --color-thresh      0                                   \
    --depth-map-scale   0.5                                 \
    --depth-cache       "$OUTPUT_DIR/front_depth_cache"     \
    --depth-cache-lru   15                                  \
    --depth-cam-batch       25                              \
    --workers               4                               \
    --image-cache-size      2

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

python -m datasets.preprocessing.helimap.helimap_preprocess_images \
    --out-root          "$OUTPUT_DIR"           \
    --splits            train val test          \
    --poses             "$CAMERA_POS"           \
    --cal-nadir         "$CAMERA_NADIR"         \
    --images-nadir      "$CAMERA_NADIR_FOLDER"  \
    --max-cam-front     0                       \
    --max-cam-nadir     6                       \
    --target-h          1040                     \
    --target-w          1040                     \
    --margin-scale      1.2                     \
    --min-crop-px       500                     \
    --min-visible       1000                     \
    --n-candidates      10                      \
    --depth-buffer      3                       \
    --depth-thresh      0.75                    \
    --color-thresh      0                       \
    --workers           5                       \
    --image-cache-size  4