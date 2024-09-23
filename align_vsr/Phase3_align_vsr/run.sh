#!/bin/bash

# 获取当前日期和时间
date=$(date "+%Y-%m-%d")
time=$(date "+%H-%M-%S")

cd /work/liuzehua/task/VSR/cnvsrc/vsr2asr/model5/Phase3_vsr2asr_v2

save_path=/work/liuzehua/task/VSR/cnvsrc/main_log/$date/$time-model5-Phase3-vsr2asr-v2-lrs2-4s-model3

CUDA_VISIBLE_DEVICES=3,4,5 python main.py save.save_path=$save_path 