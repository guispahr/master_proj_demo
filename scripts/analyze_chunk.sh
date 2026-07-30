#!/bin/bash
#SBATCH --job-name chunk_stats # Name for your job
#SBATCH --nodes 1             # do not change (unless you do MPI)
#SBATCH --nodelist lts2srv5
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 10      # reserve 4 cpus on a single node
#SBATCH --time 1000             # Runtime in minutes.
#SBATCH --mem 30000             # Reserve 10 GB RAM for the job
#SBATCH --partition cpu         # Partition to submit ('gpu' or 'cpu')
#SBATCH --qos students-msc             # QOS ('staff' or 'students')
#SBATCH --output cov-%j.txt       # Standard out goes to this file
#SBATCH --error cov-%j.txt        # Standard err goes to this file
#SBATCH --mail-user guillaume.spahr@epfl.ch     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc
#SBATCH --chdir /nfs_home/gspahr/PDM

# ACTIVATE ANACONDA
# `conda` is not on PATH on every node (e.g. the cpu partition), so source it
# explicitly before activating — otherwise `python` silently falls back to the
# system Python 2.7 and the type-annotated code fails to even parse.
if ! command -v conda >/dev/null 2>&1; then
  for _c in "$CONDA_BASE" /opt/conda "$HOME/miniconda3" "$HOME/anaconda3" /nfs_home/gspahr/miniconda3; do
    [ -n "$_c" ] && [ -f "$_c/etc/profile.d/conda.sh" ] && source "$_c/etc/profile.d/conda.sh" && break
  done
fi
eval "$(conda shell.bash hook)"
conda activate ptv3
python -c 'import sys; assert sys.version_info[0] >= 3, "conda not active: " + sys.version' || exit 1


ROOT_DIR="/net/lts2srv4//mnt/scratch/students/gspahr/ATOCHA"

python -m utils.analyze_chunks \
    --dataset atocha \
    --path "$ROOT_DIR"   \
    --out-dir "data/atocha" \
    --num-workers 8 \

ROOT_DIR="/net/lts2srv4//mnt/scratch/students/gspahr/A9CR"

python -m utils.analyze_chunks \
    --dataset a9cr \
    --path "$ROOT_DIR"   \
    --out-dir "data/19cr" \
    --num-workers 8 \


# ROOT_DIR="/mnt/scratch/students/gspahr/GridNet_preprocessed"

# python -m utils.analyze_chunks \
#     --dataset gridnet \
#     --path "$ROOT_DIR"   \
#     --out-dir "data" \