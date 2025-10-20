#!/bin/bash

export PYDEVD_WARN_EVALUATION_TIMEOUT=99999999
export PYDEVD_INTERRUPT_THREAD_TIMEOUT=99999999
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=99999999

python -m debugpy --listen 5679 --wait-for-client \
    main.py

CUDA_VISIBLE_DEVICES="0" torchrun --nnodes=1 --nproc_per_node=8 \
    train_qwenimage.py \
    --config_path configs/qwenimage_dmd.yaml \
    --logdir logs/qwenimage_dmd \
    --no_visualize \
    --disable-wandb