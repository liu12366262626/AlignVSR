#!/bin/bash

# 获取当前日期和时间
date=$(date "+%Y-%m-%d")
time=$(date "+%H-%M-%S")

cd /work/liuzehua/task/VSR/cnvsrc/vsr2asr/model5/Phase2_asr

save_path=/work/liuzehua/task/VSR/cnvsrc/main_log/$date/$time-model5-Phase2-asr

CUDA_VISIBLE_DEVICES=0,4,5 python main.py save.save_path=$save_path 