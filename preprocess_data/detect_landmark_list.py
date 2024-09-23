import argparse
import multiprocessing
import sys
#sys.path.append('/work/liuzehua/task/NCMMSC/new_one/CNVSRC2023Baseline/preparation')





def task_on_gpu(args, gpu_id, filelist):
    fl = args.list
    rank = args.rank
    shard = args.shard
    import torch
    import os
    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # 示例任务：打印CUDA设备的名称
    device = torch.device("cuda")  # 因为我们已经设置了CUDA_VISIBLE_DEVICES，所以这里的0实际上是gpu_id
    print(f"Running on {torch.cuda.get_device_name(device)}")






    
    import cv2
    import torchvision
    import sys
    #sys.path.append('/work/liuzehua/task/NCMMSC/new_one/CNVSRC2023Baseline/preparation')
    from detector import LandmarksDetector
    from video_process import VideoProcess
    import numpy as np
    import random
    import math
    from tqdm import tqdm
    import argparse
    import os
    from pathlib import Path
    import cProfile


    def load_video(data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def extract_lip(landmarks_detector, video_process, src, vfn, dst, landmark_path):
        video = load_video(f"{src}/{vfn}.mp4")
        landmarks = landmarks_detector(f"{src}/{vfn}.mp4")
        video = video_process(video, landmarks)
        torchvision.io.write_video(f"{dst}/{vfn}.mp4", video, fps=25)
        np.save(f"{landmark_path}/{vfn}.npy", landmarks)


    landmarks_detector = LandmarksDetector(device="cuda")
    video_process = VideoProcess(convert_gray=False, window_margin=1, crop_height = 128, crop_width = 128)
    psize = math.ceil(len(filelist) / shard)
    filelist = filelist[rank*psize: (rank+1)*psize]
    pbar = tqdm(total=len(filelist))
    print(f"Rank {rank} init {len(filelist)} video file")
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
    with open(f"{fl}_failed_{rank}.txt", 'a') as fp:
        fp.write('\n'.join(failed))



    

parser = argparse.ArgumentParser()
parser.add_argument('--list', default = '/work/liuzehua/task/NCMMSC/new_one/CNVSRC2023Baseline/preparation/cncvs-train.scp', help='path contains source video files')
parser.add_argument('--rank', default=0, type=int, help='index of current run')
parser.add_argument('--shard', default=2, type=int, help='size of multiprocessing pool')

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    print(f'start multiprocessing as form of {multiprocessing.get_start_method()}')
    args = parser.parse_args()
    fl = args.list
    rank = args.rank
    shard = args.shard
    
    filelist = []
    with open(fl) as fp:
        for line in fp.readlines():
            vfn, src_dir, dst_dir, landmark_dir = line.strip().split('\t')
            filelist.append((vfn, src_dir, dst_dir, landmark_dir))
    
    # 您所拥有的GPU列表
    available_gpus = [0,1,2,3,4,5]
    gpu_process_num = [0,0,0,1,3,3]
    split_num = sum(gpu_process_num)

            
    # 使用Python的多进程，为每个GPU启动两个进程
    processes = []
    last = 0
    count = 0

    task_path_list = filelist #可分成多份的任务
    for i in range(len(available_gpus)):
        for j in range(gpu_process_num[i]):

            avg = len(task_path_list) // split_num
            remain = len(task_path_list) % split_num
            size = avg + 1 if count < remain else avg
            temp_task_path_list = task_path_list[last:last + size]
            last += size
            gpu_id = available_gpus[i]
            count = count + 1
            p = multiprocessing.Process(target=task_on_gpu, args=(args, gpu_id, temp_task_path_list))
            p.start()
            processes.append(p)
            
    # 等待所有进程完成
    for p in processes:
        p.join()



