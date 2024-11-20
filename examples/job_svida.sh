#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J painn
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 20:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
##BSUB -R "span[hosts=1]"
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
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/12.4.1

/appl/cuda/12.4.1/samples/bin/x86_64/linux/release/deviceQuery

export REPO=/zhome/19/d/137388/github/DeepLearning_PAINN

# Create job_out if it is not present
if [[ ! -d ${REPO}/job_out ]]; then
	mkdir ${REPO}/job_out
fi

date=$(date +%Y%m%d_%H%M)

results_dir=${REPO}/runs/train/${date}
mkdir $results_dir

# Activate venv
#module load python3/3.11.9
source ${REPO}/venv/bin/activate

# run training
python3 ${REPO}/examples/with_validation.py \
				experiment.data.results_dir=$results_dir

