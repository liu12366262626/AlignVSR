#!/bin/bash

# 获取当前日期和时间
date=$(date "+%Y-%m-%d")
time=$(date "+%H-%M-%S")


save_path=$(dirname $(dirname $(pwd)))/main_log/$date/$time-model5-Phase2-asr

# 设置自定义路径
export CUSTOM_PATH="/work/liuzehua/task/VSR/cnvsrc"

CUDA_VISIBLE_DEVICES=0,1 python main.py save.save_path=$save_path 