#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J painn[1-2]
### -- ask for number of CPU cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
##BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -gpu "num=1"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
##BSUB -B
### -- send notification at completion--
##BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o job_out/gpu_%J_%I.out
#BSUB -e job_out/gpu_%J_%I.err
# -- end of LSF options --

module purge

nvidia-smi
# Load the cuda module
module load cuda/12.4.1

/appl/cuda/12.4.1/samples/bin/x86_64/linux/release/deviceQuery

##export REPO=/work3/mtaho/PhD/DeepLearning/DeepLearning_PAINN
export REPO=/zhome/19/d/137388/github/DeepLearning_PAINN

ARRAY1=(0.0000001 0.0000005)
rho=${ARRAY1[${LSB_JOBINDEX}-1]}
tMax=800
seed=23
target=7
echo "target: $target"

# Create a new directory for the results
date=$(date +%Y-%m-%d)
results_dir=${REPO}/runs/train/${date}/seed_$seed"_target_"$target"_rho_"$rho"_id_"$LSB_JOBINDEX
echo "results_dir: $results_dir"
mkdir -p $results_dir

# Activate venv
source ${REPO}/venv/bin/activate

# run training
python3 ${REPO}/examples/train_SAM.py \
				hydra.output_subdir=$results_dir \
				experiment.data.results_dir=$results_dir \
				experiment.data.target=$target \
				experiment.training.rho=$rho \
				experiment.training.sam_adaptive=False \
				experiment.seed=$seed \
				experiment.training.cosine_annealing_tMax=$tMax



