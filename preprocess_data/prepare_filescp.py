
import os
import random
import argparse
import sys
sys.path.append('/work/liuzehua/task/NCMMSC/new_one/CNVSRC2023Baseline/preparation')
parser = argparse.ArgumentParser()
parser.add_argument('--src', default = '/work/liuzehua/task/VTS/data/CNVSRC/multi-speaker/data_aug/one_hour_v10/others_all', help='source dir of downloaded files')
parser.add_argument('--dst', default= '/work/liuzehua/task/VTS/data/CNVSRC/multi-speaker/data_aug/one_hour_v10/preprocess_data/temp', help='dst dir of processed video files')
parser.add_argument('--dataset', default= 'cnvsrc-multi', help='which dataset to be processed')
parser.add_argument('--split', default= 'train', help='train/valid/test')
parser.add_argument('--csv_path',default='/work/liuzehua/task/NCMMSC/new_one/CNVSRC2023Baseline/my_task/temp/cnvsrc-multi/v010_aug/gen_csv/train_others_all.csv',help= 'csv_path')

args = parser.parse_args()
pwd_dir = os.path.dirname(os.path.abspath('.')) # preparation
filelist_dir = '/work/liuzehua/task/NCMMSC/new_one/CNVSRC2023Baseline/data'
csv_path = args.csv_path

def prepare_cncvs_scp(cncvs_rootdir, dst):
    filelist = []
    for split in sorted(os.listdir(cncvs_rootdir)):
        for spk in sorted(os.listdir(f"{cncvs_rootdir}/{split}")):
            if os.path.isdir(f"{cncvs_rootdir}/{split}/{spk}"):
                src_dir = os.path.join(cncvs_rootdir, split, spk, 'video')
                dst_dir = os.path.join(dst, split, spk, 'video')
                landmark_dir = os.path.join(dst, 'facelandmark')
                for fn in sorted(os.listdir(src_dir)):
                    vfn = fn[:-4]
                    filelist.append(f"{vfn}\t{src_dir}\t{dst_dir}\t{landmark_dir}")
    random.shuffle(filelist)
    with open(f'cncvs-{args.split}.scp', 'w') as fp:
        fp.write('\n'.join(filelist))
        
def prepare_ss_scp(single_speaker_rootdir, dst, split, datasetname):
    filelist = []
    # for csv in os.listdir(os.path.join(filelist_dir, 'cnvsrc-single')):
    #     if not csv.startswith(split):
    #         continue
    for line in open(csv_path).readlines():
        _, path, _, _ = line.split(',')
        vfn = single_speaker_rootdir
        dst_dir = os.path.join(dst, os.path.dirname(path))
        landmark_dir = os.path.join(dst, 'facelandmark')
        uid = os.path.basename(path)[:-4]
        #check if already preprocess
        temp_path = os.path.join(dst_dir,uid+'.mp4')
        if os.path.isfile(temp_path):
            print(f'{temp_path} is already exists')
            continue
        filelist.append(f"{uid}\t{vfn}\t{dst_dir}\t{landmark_dir}")
    random.shuffle(filelist)
    with open(f'{datasetname}-{split}.scp', 'w') as fp:
        fp.write('\n'.join(filelist))


def prepare_ms_scp(multi_speaker_rootdir, dst, split):
        filelist = []
    # for csv in os.listdir(os.path.join(filelist_dir, 'cnvsrc-multi')):
    #     if not csv.startswith(split):
    #         continue
        for line in open(csv_path).readlines():
            _, path, _, _ = line.split(',')
            #vfn = os.path.join(multi_speaker_rootdir, os.path.dirname(path))
            vfn = multi_speaker_rootdir
            dst_dir = os.path.join(dst, os.path.dirname(path))
            landmark_dir = os.path.join(dst, 'facelandmark')
            uid = os.path.basename(path)[:-4]
            #check if already preprocess
            temp_path = os.path.join(dst_dir,uid+'.mp4')
            if os.path.isfile(temp_path):
                print(f'{temp_path} is already exists')
                continue
            filelist.append(f"{uid}\t{vfn}\t{dst_dir}\t{landmark_dir}")
        random.shuffle(filelist)
        with open(f'cnvsrc-multi-{split}.scp', 'w') as fp:
            fp.write('\n'.join(filelist))
    

if __name__== '__main__':
    args = parser.parse_args()
    src = args.src
    dst = args.dst
    dataset = args.dataset
    split = args.split

    prepare_ss_scp(src, dst, split, dataset)
    
    # if dataset == 'cncvs':
    #     prepare_cncvs_scp(f'{src}/cncvs', f'{dst}')
    # elif dataset == 'cnvsrc-single':
    #     prepare_ss_scp(src, f'{dst}/cnvsrc-single/', split)
    # elif dataset == 'cnvsrc-multi':
    #     prepare_ms_scp(src, f'{dst}/cnvsrc-multi/', split)
    

