import argparse
import multiprocessing
import torch
import torchvision
import os
import numpy as np
import math
from tqdm import tqdm
from pathlib import Path
from detector import LandmarksDetector
from video_process import VideoProcess



def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess video dataset")
    parser.add_argument('--root_dir', type=str, required=True, help="Path to the input video dataset (LRS2)")
    parser.add_argument('--dst_path', type=str, required=True, help="Path to save preprocessed data")
    return parser.parse_args()


def task_on_gpu(gpu_id, filelist):
    rank = 0
    shard = 1

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda")  
    print(f"Running on {torch.cuda.get_device_name(device)}")

    def load_video(data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def extract_lip(landmarks_detector, video_process, src, vfn, dst, landmark_path):
        video = load_video(f"{src}/{vfn}.mp4")
        landmarks = landmarks_detector(f"{src}/{vfn}.mp4")
        video = video_process(video, landmarks)
        torchvision.io.write_video(f"{dst}/{vfn}.mp4", video, fps=25)
        np.save(f"{landmark_path}/{vfn}.npy", landmarks)

    landmarks_detector = LandmarksDetector(device="cuda")
    video_process = VideoProcess(convert_gray=False, window_margin=1, crop_height=96, crop_width=96)
    psize = math.ceil(len(filelist) / shard)
    filelist = filelist[rank * psize: (rank + 1) * psize]
    pbar = tqdm(total=len(filelist))
    print(f"Rank {gpu_id} init {len(filelist)} video file")
    pbar.set_description(f'rank {rank}')
    res = []
    failed = []

    for vfn, src_dir, dst_dir, landmark_dir in filelist:
        try:
            Path(dst_dir).mkdir(exist_ok=True, parents=True)
            Path(landmark_dir).mkdir(exist_ok=True, parents=True)
            extract_lip(landmarks_detector, video_process, src_dir, vfn, dst_dir, landmark_dir)
            res.append(True)
        except Exception as e:
            print(f"{vfn} failed")
            failed.append(vfn)
        pbar.update()

    print(f"Rank {rank} finish {len(filelist)} video file")
    with open(f"failed_{gpu_id}.txt", 'a') as fp:
        fp.write('\n'.join(failed))


def load_data(root_dir, dst_path):
    mp4_exist = []
    for subdir, _, files in os.walk(dst_path):
        for file in files:
            if file.lower().endswith(".mp4"):
                mp4_exist.append(file)

    mp4_files = []
    filelist = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".mp4"):
                mp4_files.append(os.path.join(subdir, file))

    for mp4_file in mp4_files:
        if os.path.basename(mp4_file) in mp4_exist:
            continue
        uid = os.path.basename(mp4_file)[:-4]
        vfn = os.path.dirname(mp4_file)
        dst_dir = dst_path + '/data' + vfn.replace(root_dir, '')
        landmark_dir = dst_path + '/facelandmark' + vfn.replace(root_dir, '')
        filelist.append((uid, vfn, dst_dir, landmark_dir))

    print(f'{len(filelist)} need to process, total {len(mp4_files)}')
    return filelist


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()

    # 打印传入的参数
    print(f'Root directory: {args.root_dir}')
    print(f'Destination path: {args.dst_path}')

    multiprocessing.set_start_method('spawn', force=True)
    print(f'start multiprocessing as form of {multiprocessing.get_start_method()}')

    # 使用命令行输入的路径
    root_dir = args.root_dir
    dst_path = args.dst_path

    # 加载数据
    filelist = load_data(root_dir, dst_path)

    # GPU list for example (6 GPUS)
    available_gpus = [0, 1, 2, 3, 4, 5]
    # the number of multiprocessing on single GPU (for example 1 process in GPU_ID 0)
    gpu_process_num = [1, 0, 0, 0, 0, 0]
    split_num = sum(gpu_process_num)

    processes = []
    last = 0
    count = 0

    task_path_list = filelist
    for i in range(len(available_gpus)):
        for j in range(gpu_process_num[i]):

            avg = len(task_path_list) // split_num
            remain = len(task_path_list) % split_num
            size = avg + 1 if count < remain else avg
            temp_task_path_list = task_path_list[last:last + size]
            last += size
            gpu_id = available_gpus[i]
            count = count + 1
            p = multiprocessing.Process(target=task_on_gpu, args=(gpu_id, temp_task_path_list))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()
