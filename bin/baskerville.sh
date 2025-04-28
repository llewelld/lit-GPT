#!/bin/bash
#SBATCH --qos turing
#SBATCH --account  vjgo8416-ml-workload
#SBATCH --time 0:30:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 36
#SBATCH --job-name mingpt-test

# Execute using:
# sbatch -o stdout-%A_%a.out --array=1-2 ./scripts/batch.sh
# Copied from: https://github.com/alan-turing-institute/rcp-benchmark/blob/dawn/projects/litgpt/scripts/baskerville/batch.sh

module purge
module load baskerville

echo MinGPT Batch run
echo ============================================
echo SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}
echo SLURM_JOB_ID: ${SLURM_JOB_ID}
echo SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}
echo SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}
echo SLURM_ARRAY_TASK_COUNT: ${SLURM_ARRAY_TASK_COUNT}
echo SLURM_ARRAY_TASK_MAX: ${SLURM_ARRAY_TASK_MAX}
echo SLURM_ARRAY_TASK_MIN: ${SLURM_ARRAY_TASK_MIN}
echo ============================================

module restore system
module load Python/3.10.4-GCCcore-11.3.0
module load GCC/11.3.0
module load CUDA/11.7.0

# Define the path to your existing Conda environment (modify as appropriate)
VENV_PATH="./venv-test"

pushd /bask/homes/o/ovau2564/vjgo8416-ml-workload/rcp-benchmark/projects/litgpt/lit-GPT

echo "Creating virtual environment"
if [ -d "$VENV_PATH" ]; then
  echo "Virtual environment exists, activating"
  # Activate the environment
  echo "Activating virtual environment"
  . "${VENV_PATH}"/bin/activate
else
  echo "Creating virtual environment"
  python3 -m venv "${VENV_PATH}"
  # Activate the environment
  echo "Activating virtual environment"
  . "${VENV_PATH}"/bin/activate
  echo "Installing requirements"
  pip install pip --upgrade
  pip install -e .
  pip install -r requirements.txt
  pip install -r requirements/nanogpt.txt
fi

echo
echo "######################################"
echo "Batch configuration"
echo "######################################"
echo

CONFIG_GPUS=1
CONFIG_LAYERS=12
CONFIG_HEAD=12
CONFIG_EMBD=768
CONFIG_PRECISION=16
CONFIG_BATCH_SIZE=$((2**(5+${SLURM_ARRAY_TASK_ID})))
CONFIG_NUM_WORKERS=36
CONFIG_STRATEGY=ddp

echo "gpus:" ${CONFIG_GPUS}
echo "Batch number: ${SLURM_ARRAY_TASK_ID}"
echo "n_layer: ${CONFIG_LAYERS}"
echo "n_head: ${CONFIG_HEAD}"
echo "n_embd: ${CONFIG_EMBD}"
echo "precision: ${CONFIG_PRECISION}"
echo "batch_size: ${CONFIG_BATCH_SIZE}"
echo "num_workers: ${CONFIG_NUM_WORKERS}"
echo "strategy: ${CONFIG_STRATEGY}"

echo
echo "######################################"
echo "Starting"
echo "######################################"
echo

# Track GPU metrics
stdbuf -o0 nvidia-smi dmon -o TD -s puct -d 1 > "dmon-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}".txt &

# Track CPU metrics
stdbuf -o0 vmstat -t 1 -y > "cpu-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}".txt &

srun python3 train.py \
	--devices ${CONFIG_GPUS} \
	--n_layer ${CONFIG_LAYERS} \
	--n_head ${CONFIG_HEAD} \
	--n_embd ${CONFIG_EMBD} \
	--precision ${CONFIG_PRECISION} \
	--batch_size ${CONFIG_BATCH_SIZE} \
	--num_workers ${CONFIG_NUM_WORKERS} \
	--strategy ${CONFIG_STRATEGY} \
	--max_epochs 2 \
	--implementation mingpt \
	--model_type None \
	--enable_progress_bar 0

echo
echo "######################################"
echo "Done"
echo "######################################"
echo

echo "Deactivating virtual environment"
deactivate
popd
