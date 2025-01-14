#!/bin/bash
#SBATCH --job-name=building_ccd
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=30GB
#SBATCH -A raiselab
#SBATCH --array=1-216
#SBATCH --partition="standard"
#SBATCH --error=error/building_ccd_%A_%a.err
#SBATCH --output=slurm-save/building_ccd_%A_%a.out


OPTS=$(sed -n "${SLURM_ARRAY_TASK_ID}"p building_ccd.in)

echo "$OPTS"

read -ra values <<< "$OPTS"

if [[ "${#values[@]}" -eq  8 ]]; then
    a="${values[0]}"
    b="${values[1]}"
    c="${values[2]}"
    d="${values[3]}"
    e="${values[4]}"
    f="${values[5]}"
    g="${values[6]}"
    h="${values[7]}"


    conda_env_name=BLO
    conda_env_path=/home/jk4pn/.conda/envs/$conda_env_name


    module purge
    module load ffmpeg
    module load gcc
    module load ninja
    module load miniforge
    # module load glfw
    source activate $conda_env_name
    export PATH="$conda_env_path/bin:$PATH"
    export LD_LIBRARY_PATH="$conda_env_path/lib:$LD_LIBRARY_PATH"

    python /home/jk4pn/COCO/multizone_building_control_codesign.py --epochs 10 --ntrain "$a" --ntest "$b" --lr "$c" --nsteps "$d" --nzones "$e" --penalty "$f" --alpha "$g" --index "$h"
else
    echo "Failed: Unexpected number of values."
    # Place the failure code or exit here.
    exit 1
fi
