#!/bin/bash

# get current time
date=$(date "+%Y-%m-%d")
time=$(date "+%H-%M-%S")

code_root_dir=$(dirname $(dirname $(pwd)))

save_path=$code_root_dir/main_log/$date/$time-Phase2-asr


# set root
export CUSTOM_PATH=$code_root_dir

CUDA_VISIBLE_DEVICES=0,1 python main.py save.save_path=$save_path \
                        code_root_dir=$code_root_dir\
                        audio_data_root_dir= \
                        csv_name = \
                        hubert_model = \
                        k_means_model = \
                        data.dataset.label_dir = \
                        data.dataset.val_file = \

#exmaple
# CUDA_VISIBLE_DEVICES=0,1 python main.py save.save_path=$save_path \
#                         code_root_dir=$code_root_dir\
#                         audio_data_root_dir=/work/liuzehua/task/VSR/data/LRS/LRS2-BBC\
#                         csv_name =train \
#                         hubert_model =/work/liuzehua/task/VSR/cnvsrc/vsr2asr/model5/English-hubert-large \
#                         k_means_model =/work/liuzehua/task/VSR/cnvsrc/vsr2asr/model5/Phase1_k-means_cluster/kmeans_model.joblib \
#                         data.dataset.label_dir =/work/liuzehua/task/VSR/cnvsrc/data/vsr2asr/model5/Phase2/LRS2 \
#                         data.dataset.val_file =test.csv \