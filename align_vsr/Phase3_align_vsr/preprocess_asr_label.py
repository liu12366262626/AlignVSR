import multiprocessing
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tqdm import tqdm
import csv





def task_on_gpu(gpu_id, path_list, task_id, return_dict, audio_data_root, video_data_root):
    import os
    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    import sys
    sys.path.append('/work/liuzehua/task/VSR/cnvsrc')#设置一些导入包的原路径
    # 示例任务：打印CUDA设备的名称
    device = torch.device("cuda:0")  # 因为我们已经设置了CUDA_VISIBLE_DEVICES，所以这里的0实际上是gpu_id
    print(f"Running on {torch.cuda.get_device_name(device)}")
    from transformers import (
        Wav2Vec2FeatureExtractor,
        HubertModel
    )
    from joblib import load
    import soundfile as sf
    import torchvision
    from vsr2asr.model5.Phase3_vsr2asr_v2.transforms import VideoTransform
    from tqdm import tqdm



    hubert_model_path = '/work/liuzehua/task/VSR/cnvsrc/vsr2asr/model5/English-hubert-large'
    k_means_path = '/work/liuzehua/task/VSR/cnvsrc/vsr2asr/model5/Phase1_k-means_cluster/kmeans_model.joblib'
    device = 'cuda'

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_path)
    hubert_model = HubertModel.from_pretrained(hubert_model_path)
    hubert_model.to(device)
    kmeans_model = load(k_means_path)
    video_transform=VideoTransform("train")

    def cut_or_pad(data, size, dim=0):
        #42864 + 16 = 42880
        """
        Pads or trims the data along a dimension.
        """
        if data.size(dim) < size:
            padding = size - data.size(dim)
            data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
        elif data.size(dim) > size:
            data = data[:size]
        assert data.size(dim) == size
        
        return data

    def load_video(path):
        """
        rtype: torch, T x C x H x W
        """
        vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
        vid = vid.permute((0, 3, 1, 2))
        return vid

    csv_data = []


    for row in tqdm(path_list, desc='Processing CSV', unit=' row', total = len(path_list)):


        dataset_name, video_rel_path, audio_rel_path, input_length, token_id  = row
        video_path = os.path.join(video_data_root, dataset_name, video_rel_path)
        audio_path = os.path.join(audio_data_root, dataset_name, audio_rel_path)
        # 判断video_path和audio_path是否同时存在
        if not (os.path.exists(video_path) and os.path.exists(audio_path)):
            continue

        video = load_video(video_path)
        video = video_transform(video)

        rate_ratio=640
        wav, sr = sf.read(audio_path)
        input_values = feature_extractor(wav, return_tensors="pt", sampling_rate = sr).input_values
        input_values = cut_or_pad(input_values.transpose(1,0), len(video) * rate_ratio, )
        input_values = input_values.transpose(1,0)
        input_values = input_values.to(device)
        outputs = hubert_model(input_values).last_hidden_state

        # Reshape x to (batch_size * time_steps, embedding_size)
        batch_size, time_steps, embedding_size = outputs.size()
        x_reshaped = outputs.view(batch_size * time_steps, embedding_size).detach().cpu()

        # Predict cluster center indices for each sample using the k-means model
        labels = kmeans_model.predict(x_reshaped)

        # Convert labels to a PyTorch tensor and ensure it is on the same device as the original tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=outputs.device).view(batch_size, time_steps)
        labels_tensor = labels_tensor.squeeze(0)
        # labels_tensor = torch.randint(low=0, high=200, size=(random.randint(0, 199),))
        csv_data.append([dataset_name, video_rel_path, audio_rel_path, input_length, token_id, labels_tensor.tolist()])

    return_dict[task_id] = csv_data

    # # 将 data 写入到 CSV 文件中
    # with open(output_csv_path, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     for row in csv_data:
    #         writer.writerow(row)






if __name__ == '__main__': 
    multiprocessing.set_start_method('spawn', force=True)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    csv_path = '/work/liuzehua/task/VSR/cnvsrc/data/vsr2asr/model5/Phase3/LRS2/temp/train.csv'
    audio_data_root = '/work/liuzehua/task/VSR/data/LRS/LRS2-BBC'
    video_data_root = '/work/liuzehua/task/VSR/data/LRS/LRS2-BBC'
    save_path = '/work/liuzehua/task/VSR/cnvsrc/data/vsr2asr/model5/Phase3/LRS2'
    file_name = os.path.basename(csv_path)
    output_csv_path = os.path.join(save_path, file_name)
    error_path = os.path.join(save_path, f'{file_name}_error.txt')

    task_path_list = []  # 可分成多份的任务
    errors = []  # 存储错误信息

    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in tqdm(reader, desc='Processing CSV', unit=' row'):
            dataset_name, video_rel_path, audio_rel_path, input_length, token_id = row
            video_path = os.path.join(video_data_root, dataset_name, video_rel_path)
            audio_path = os.path.join(audio_data_root, dataset_name, audio_rel_path)
            
            # 判断video_path和audio_path是否同时存在
            if not os.path.exists(video_path):
                errors.append(f"Video path not found: {video_path}")
                continue
            if not os.path.exists(audio_path):
                errors.append(f"Audio path not found: {audio_path}")
                continue
            
            task_path_list.append(row)

    # 将错误信息写入到error.txt文件
    with open(error_path, 'w') as error_file:
        for error in errors:
            error_file.write(f"{error}\n")


    # 您所拥有的GPU列表
    available_gpus = [0,1,2,3,4,5]
    gpu_process_num = [0,2,1,1,0,0]
    split_num = sum(gpu_process_num)

            
    # 使用Python的多进程，为每个GPU启动两个进程
    processes = []
    last = 0
    count = 0

    for i in range(len(available_gpus)):
        for j in range(gpu_process_num[i]):

            avg = len(task_path_list) // split_num
            remain = len(task_path_list) % split_num
            size = avg + 1 if count < remain else avg
            temp_path_list = task_path_list[last:last + size]
            last += size
            gpu_id = available_gpus[i]
            count = count + 1
            p = multiprocessing.Process(target=task_on_gpu, args=(gpu_id, temp_path_list, count, return_dict, audio_data_root, video_data_root))
            p.start()
            processes.append(p)
            
    # 等待所有进程完成
    for p in processes:
        p.join()


    # 将所有的列表合并成一个大列表
    merged_list = []
    for lst in return_dict.values():
        merged_list.extend(lst)


    # 将 data 写入到 CSV 文件中
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in merged_list:
            writer.writerow(row)