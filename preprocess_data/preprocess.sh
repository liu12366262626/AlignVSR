# fill the blank and run
DOWNLOAD_DATA_PATH='/work/lixiaolou/data/CNVSRC/CNVSRC2023/cnvsrc-single/dev/video'
TARGET_DATA_PATH='/work/liuzehua/task/VSR/data/preprocess/cnvsrc-single_128' #TARGET_DATA_PATH='/work/liuzehua/task/NCMMSC/new_one/CNVSRC2023Baseline/result1/CNVSRC_lips'
DATASET_NAME='cnvsrc-single_128'  #用来选择哪个数据集进行预处理
SPLIT='all'
CSV_PATH='/work/liuzehua/task/VSR/cnvsrc/data/cnvsrc-single/vsr/train_and_valid.csv' #csv file includes data that needs preprocess
CODE_ROOT_PATH=$(dirname "$PWD")


if test -z "$DOWNLOAD_DATA_PATH"; then 
echo "DOWNLOAD_DATA_PATH is not set!"
exit 0
fi
if test -z "$TARGET_DATA_PATH"; then 
echo "TARGET_DATA_PATH is not set!"
exit 0
fi
if test -z "$DATASET_NAME"; then 
echo "DATASET_NAME is not set!"
exit 0
fi
if test -z "$SPLIT"; then 
echo "SPLIT is not set!"
exit 0
fi

python prepare_filescp.py \
   --src $DOWNLOAD_DATA_PATH \
   --dst $TARGET_DATA_PATH \
   --dataset $DATASET_NAME \
   --split $SPLIT \
   --csv_path $CSV_PATH

python detect_landmark_list.py --list $DATASET_NAME-$SPLIT.scp --rank 0 --shard 1
