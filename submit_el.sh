#!/bin/bash -l
#PBS -N white_box_fid
#PBS -l ncpus=10
#PBS -l ngpus=1
#PBS -l mem=20gb
#PBS -l cpuarch=avx512
#PBS -l walltime=168:00:00

cd $PBS_O_WORKDIR
echo "queue started"

module purge
echo "modules purged"

#python/3.7.2-foss-2018a
#module load python/3.7.2-foss-2018a
#module load cuda/10.1.243-gcc-8.3.0
#module load cudnn/7.6.4.38-gcccuda-2019b
#echo "cuda loaded"
#module load tensorflow/2.3.1-fosscuda-2019b-python-3.7.4
#echo "tensorflow loaded"
#module load scikit-learn/0.18.1-foss-2016b-python-3.5.2
#echo "sklearn loaded"

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/n9455647/three-phase-fidelity-wip/hpc_venv/lib

module load anaconda3
conda activate conda_test

#source hpc_venv/bin/activate
python -c "import sys; print('Executable path', sys.executable)"
#python -m pip install imodels
#python -m pip show imodels
#python -m pip freeze > requirements.txt

echo "starting test"
python3 el_white_box_fidelity.py $dataset $bucketing $encoding $model $xai
echo "test ended"
echo $dataset $bucketing $encoding $model $xai

conda deactivate
module unload anaconda3
