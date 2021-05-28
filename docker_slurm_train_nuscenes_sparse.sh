 #!/bin/bash
srun --gpus=1\
 --mem-per-gpu=32G\
 --cpus-per-gpu=8\
 --qos=eaton-high\
 --container-mounts=/Datasets:/Datasets,/home/kvedder/code/second.pytorch:/second.pytorch\
 --container-image=docker-registry.grasp.cluster#second-pytorch-sparse\
 --time=72:00:00\
 --partition=eaton-compute\
 bash -c "nvidia-smi && python ./pytorch/train.py train --config_path=./configs/nuscenes/all.pp.largea_sparse.config --model_dir=./model_dir_nuscenes_car_16_sparse_actually/"
# -w node-3090-1\
# --pty\
# rl_algs/LPG-FTW/experiments/habitat_pgftw.py
#create_data.py nuscenes_data_prep ROOT_PATH VERSION DATASET_NAME <flags>
#  optional flags:        --max_sweeps
