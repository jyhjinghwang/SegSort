#!/bin/bash
# This script is used for training, inference and benchmarking
# the PSPNet with SegSort on PASCAL VOC 2012.
#
# Usage:
#   # From SegSort/ directory.
#   bash bashscripts/voc12/train_segsort.sh


# Set up training hyper-parameters.
BATCH_SIZE=8
TRAIN_INPUT_SIZE=336,336
WEIGHT_DECAY=5e-4
ITER_SIZE=1
NUM_STEPS=30000
NUM_CLASSES=21
LEARNING_RATE=2e-3

# Set up parameters for inference.
INFERENCE_INPUT_SIZE=480,480
INFERENCE_STRIDES=320,320
INFERENCE_SPLIT=val

# Set up SegSort hyper-parameters.
CONCENTRATION=10
NUM_BANKS=2
EMBEDDING_DIM=32
NUM_CLUSTERS=5
KMEANS_ITERATIONS=10
K_IN_NEAREST_NEIGHBORS=15

# Set up path for saving models.
SNAPSHOT_DIR=snapshots/voc12/segsort/segsort_lr2e-3_it30000

# Set up the procedure pipeline.
IS_TRAIN_1=1
IS_PROTOTYPE_1=1
IS_INFERENCE_1=1
IS_INFERENCE_MSC_1=0
IS_BENCHMARK_1=1
IS_TRAIN_2=1
IS_PROTOTYPE_2=1
IS_INFERENCE_2=1
IS_INFERENCE_MSC_2=1
IS_BENCHMARK_2=1

# Update PYTHONPATH.
export PYTHONPATH=`pwd`:$PYTHONPATH

# Set up the data directory.
DATAROOT=/ssd/jyh/datasets

# Train for the 1st stage.
if [ ${IS_TRAIN_1} -eq 1 ]; then
  python3 pyscripts/train/train_segsort.py\
    --snapshot_dir ${SNAPSHOT_DIR}/stage1\
    --restore_from snapshots/imagenet/trained/resnet_v1_101.ckpt\
    --data_list dataset/voc12/train+.txt\
    --data_dir ${DATAROOT}/VOCdevkit/\
    --batch_size ${BATCH_SIZE}\
    --save_pred_every ${NUM_STEPS}\
    --update_tb_every 50\
    --input_size ${TRAIN_INPUT_SIZE}\
    --learning_rate ${LEARNING_RATE}\
    --weight_decay ${WEIGHT_DECAY}\
    --iter_size ${ITER_SIZE}\
    --num_classes ${NUM_CLASSES}\
    --num_steps $(($NUM_STEPS+1))\
    --concentration ${CONCENTRATION}\
    --num_banks ${NUM_BANKS}\
    --embedding_dim ${EMBEDDING_DIM}\
    --num_clusters ${NUM_CLUSTERS}\
    --kmeans_iterations ${KMEANS_ITERATIONS}\
    --random_mirror\
    --random_scale\
    --random_crop\
    --not_restore_classifier\
    --is_training
fi

# Prototype for the 1st stage.
if [ ${IS_PROTOTYPE_1} -eq 1 ]; then
  python3 pyscripts/inference/extract_prototypes.py\
    --data_dir ${DATAROOT}/VOCdevkit/\
    --data_list dataset/voc12/train+.txt\
    --restore_from ${SNAPSHOT_DIR}/stage1/model.ckpt-${NUM_STEPS}\
    --input_size ${INFERENCE_INPUT_SIZE}\
    --strides ${INFERENCE_STRIDES}\
    --num_classes ${NUM_CLASSES}\
    --ignore_label 255\
    --embedding_dim ${EMBEDDING_DIM}\
    --num_clusters ${NUM_CLUSTERS}\
    --kmeans_iterations ${KMEANS_ITERATIONS}\
    --save_dir ${SNAPSHOT_DIR}/stage1/results/train+
fi

# Single-scale inference for the 1st stage.
if [ ${IS_INFERENCE_1} -eq 1 ]; then
  python3 pyscripts/inference/inference_segsort.py\
    --data_dir ${DATAROOT}/VOCdevkit/\
    --data_list dataset/voc12/${INFERENCE_SPLIT}.txt\
    --input_size 720,720\
    --strides ${INFERENCE_STRIDES}\
    --restore_from ${SNAPSHOT_DIR}/stage1/model.ckpt-${NUM_STEPS}\
    --colormap misc/colormapvoc.mat\
    --num_classes ${NUM_CLASSES}\
    --ignore_label 255\
    --embedding_dim ${EMBEDDING_DIM}\
    --num_clusters ${NUM_CLUSTERS}\
    --kmeans_iterations ${KMEANS_ITERATIONS}\
    --k_in_nearest_neighbors ${K_IN_NEAREST_NEIGHBORS}\
    --save_dir ${SNAPSHOT_DIR}/stage1/results/${INFERENCE_SPLIT}\
    --prototype_dir ${SNAPSHOT_DIR}/stage1/results/train+/prototypes
fi

# Multi-scale inference for the 1st stage.
if [ ${IS_INFERENCE_MSC_1} -eq 1 ]; then
  python3 pyscripts/inference/inference_segsort_msc.py\
    --data_dir ${DATAROOT}/VOCdevkit/\
    --data_list dataset/voc12/${INFERENCE_SPLIT}.txt\
    --input_size ${INFERENCE_INPUT_SIZE}\
    --strides ${INFERENCE_STRIDES}\
    --restore_from ${SNAPSHOT_DIR}/stage1/model.ckpt-${NUM_STEPS}\
    --colormap misc/colormapvoc.mat\
    --num_classes ${NUM_CLASSES}\
    --ignore_label 255\
    --flip_aug\
    --scale_aug\
    --embedding_dim ${EMBEDDING_DIM}\
    --num_clusters ${NUM_CLUSTERS}\
    --kmeans_iterations ${KMEANS_ITERATIONS}\
    --k_in_nearest_neighbors ${K_IN_NEAREST_NEIGHBORS}\
    --save_dir ${SNAPSHOT_DIR}/stage1/results/${INFERENCE_SPLIT}\
    --prototype_dir ${SNAPSHOT_DIR}/stage1/results/train+/prototypes
fi

# Benchmark for the 1st stage.
if [ ${IS_BENCHMARK_1} -eq 1 ]; then
  python3 pyscripts/benchmark/benchmark_by_mIoU.py\
    --pred_dir ${SNAPSHOT_DIR}/stage1/results/${INFERENCE_SPLIT}/gray/\
    --gt_dir ${DATAROOT}/VOCdevkit/VOC2012/segcls/\
    --num_classes ${NUM_CLASSES}
fi


LEARNING_RATE=2e-4
# Train for the 2nd stage.
if [ ${IS_TRAIN_2} -eq 1 ]; then
  python3 pyscripts/train/train_segsort.py\
    --snapshot_dir ${SNAPSHOT_DIR}/stage2\
    --restore_from ${SNAPSHOT_DIR}/stage1/model.ckpt-30000\
    --data_list dataset/voc12/train.txt\
    --data_dir ${DATAROOT}/VOCdevkit/\
    --batch_size ${BATCH_SIZE}\
    --save_pred_every ${NUM_STEPS}\
    --update_tb_every 50\
    --input_size ${TRAIN_INPUT_SIZE}\
    --learning_rate ${LEARNING_RATE}\
    --weight_decay ${WEIGHT_DECAY}\
    --iter_size ${ITER_SIZE}\
    --num_classes ${NUM_CLASSES}\
    --num_steps $(($NUM_STEPS+1))\
    --concentration ${CONCENTRATION}\
    --num_banks ${NUM_BANKS}\
    --embedding_dim ${EMBEDDING_DIM}\
    --num_clusters ${NUM_CLUSTERS}\
    --kmeans_iterations ${KMEANS_ITERATIONS}\
    --random_mirror\
    --random_scale\
    --random_crop\
    --not_restore_classifier\
    --is_training
fi

# Prototype for the 1st stage.
if [ ${IS_PROTOTYPE_2} -eq 1 ]; then
  python3 pyscripts/inference/extract_prototypes.py\
    --data_dir ${DATAROOT}/VOCdevkit/\
    --data_list dataset/voc12/train.txt\
    --restore_from ${SNAPSHOT_DIR}/stage2/model.ckpt-${NUM_STEPS}\
    --input_size ${INFERENCE_INPUT_SIZE}\
    --strides ${INFERENCE_STRIDES}\
    --num_classes ${NUM_CLASSES}\
    --ignore_label 255\
    --embedding_dim ${EMBEDDING_DIM}\
    --num_clusters ${NUM_CLUSTERS}\
    --kmeans_iterations ${KMEANS_ITERATIONS}\
    --save_dir ${SNAPSHOT_DIR}/stage2/results/train
fi

# Single-scale inference for the 2nd stage.
if [ ${IS_INFERENCE_2} -eq 1 ]; then
  python3 pyscripts/inference/inference_segsort.py\
    --data_dir ${DATAROOT}/VOCdevkit/\
    --data_list dataset/voc12/${INFERENCE_SPLIT}.txt\
    --input_size 720,720\
    --strides ${INFERENCE_STRIDES}\
    --restore_from ${SNAPSHOT_DIR}/stage2/model.ckpt-${NUM_STEPS}\
    --colormap misc/colormapvoc.mat\
    --num_classes ${NUM_CLASSES}\
    --ignore_label 255\
    --embedding_dim ${EMBEDDING_DIM}\
    --num_clusters ${NUM_CLUSTERS}\
    --kmeans_iterations ${KMEANS_ITERATIONS}\
    --k_in_nearest_neighbors ${K_IN_NEAREST_NEIGHBORS}\
    --save_dir ${SNAPSHOT_DIR}/stage2/results/${INFERENCE_SPLIT}\
    --prototype_dir ${SNAPSHOT_DIR}/stage2/results/train/prototypes
fi

# Multi-scale inference for the 2nd stage.
if [ ${IS_INFERENCE_MSC_2} -eq 1 ]; then
  python3 pyscripts/inference/inference_segsort_msc.py\
    --data_dir ${DATAROOT}/VOCdevkit/\
    --data_list dataset/voc12/${INFERENCE_SPLIT}.txt\
    --input_size ${INFERENCE_INPUT_SIZE}\
    --strides ${INFERENCE_STRIDES}\
    --restore_from ${SNAPSHOT_DIR}/stage2/model.ckpt-${NUM_STEPS}\
    --colormap misc/colormapvoc.mat\
    --num_classes ${NUM_CLASSES}\
    --ignore_label 255\
    --flip_aug\
    --scale_aug\
    --embedding_dim ${EMBEDDING_DIM}\
    --num_clusters ${NUM_CLUSTERS}\
    --kmeans_iterations ${KMEANS_ITERATIONS}\
    --k_in_nearest_neighbors ${K_IN_NEAREST_NEIGHBORS}\
    --save_dir ${SNAPSHOT_DIR}/stage2/results/${INFERENCE_SPLIT}\
    --prototype_dir ${SNAPSHOT_DIR}/stage2/results/train/prototypes
fi

# Benchmark for the 2nd stage.
if [ ${IS_BENCHMARK_2} -eq 1 ]; then
  python3 pyscripts/benchmark/benchmark_by_mIoU.py\
    --pred_dir ${SNAPSHOT_DIR}/stage2/results/${INFERENCE_SPLIT}/gray/\
    --gt_dir ${DATAROOT}/VOCdevkit/VOC2012/segcls/\
    --num_classes ${NUM_CLASSES}
fi
