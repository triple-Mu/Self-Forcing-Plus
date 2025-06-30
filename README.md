<p align="center">
<h1 align="center">Self Forcing</h1>
<h3 align="center">Bridging the Train-Test Gap in Autoregressive Video Diffusion</h3>
</p>
<p align="center">
  <p align="center">
    <a href="https://www.xunhuang.me/">Xun Huang</a><sup>1</sup>
    ·
    <a href="https://zhengqili.github.io/">Zhengqi Li</a><sup>1</sup>
    ·
    <a href="https://guandehe.github.io/">Guande He</a><sup>2</sup>
    ·
    <a href="https://mingyuanzhou.github.io/">Mingyuan Zhou</a><sup>2</sup>
    ·
    <a href="https://research.adobe.com/person/eli-shechtman/">Eli Shechtman</a><sup>1</sup><br>
    <sup>1</sup>Adobe Research <sup>2</sup>UT Austin
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2506.08009">Paper</a> | <a href="https://self-forcing.github.io">Website</a> | <a href="https://huggingface.co/gdhe17/Self-Forcing/tree/main">Models (HuggingFace)</a></h3>
</p>

---

Self Forcing trains autoregressive video diffusion models by **simulating the inference process during training**, performing autoregressive rollout with KV caching. It resolves the train-test distribution mismatch and enables **real-time, streaming video generation on a single RTX 4090** while matching the quality of state-of-the-art diffusion models.

---


<table>
  <tr>
    <td align="center">
      <video src="https://github.com/GoatWu/Self-Forcing-Plus/blob/main/demos/output_lightx2v_wan_t2v_t06.mp4" width="100%"></video>
    </td>
    <td align="center">
      <video src="https://github.com/GoatWu/Self-Forcing-Plus/blob/main/demos/output_lightx2v_wan_t2v_t01.mp4" width="100%"></video>
    </td>
    <td align="center">
      <video src="https://github.com/GoatWu/Self-Forcing-Plus/blob/main/demos/output_lightx2v_wan_t2v_t03.mp4" width="100%"></video>
    </td>
  </tr>
</table>


## Requirements
We tested this repo on the following setup:
* Nvidia GPU with at least 24 GB memory (RTX 4090, A100, and H100 are tested).
* Linux operating system.
* 64 GB RAM.

Other hardware setup could also work but hasn't been tested.

## Installation
Create a conda environment and install dependencies:
```
conda create -n self_forcing python=3.10 -y
conda activate self_forcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

## Quick Start
### Download checkpoints
```
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir wan_models/Wan2.1-T2V-14B
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir wan_models/Wan2.1-I2V-14B-480P
```

## T2V Training

DMD training for bidirectional models do not need ODE initialization.

### DataSet Preparation

We build the dataset in the following way, each file contains a single prompt:

```
data_folder
  |__1.txt
  |__2.txt
  ...
  |__xxx.txt
```

### DMD Training
```
torchrun --nnodes=8 --nproc_per_node=8 \
--rdzv_id=5235 \
--rdzv_backend=c10d \
--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
train.py \
--config_path configs/self_forcing_14b_dmd.yaml \
--logdir logs/self_forcing_14b_dmd \
--no_visualize \
--disable-wandb
```

Our training run uses 3000 iterations and completes in under 3 days using 64 H100 GPUs.

## I2V-480P Training

### DataSet Preparation

1. Generate a series of videos using the original Wan2.1 model.

2. Generate the VAE latents.
```bash
python scripts/compute_vae_latent.py \
--input_video_folder {video_folder} \
--output_latent_folder {latent_folder} \
--model_name Wan2.1-T2V-14B \
--prompt_folder {prompt_folder}
```

3. Separate the first frame of the videos and create an lmdb dataset.
```bash
python scripts/create_lmdb_14b_shards.py \
--data_path {latent_folder} \
--prompt_path {prompt_folder} \
--lmdb_path {lmdb_folder}
```

### DMD Training
```
torchrun --nnodes=8 --nproc_per_node=8 \
--rdzv_id=5235 \
--rdzv_backend=c10d \
--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
train.py \
--config_path configs/self_forcing_14b_i2v_dmd.yaml \
--logdir logs/self_forcing_14b_i2v_dmd \
--no_visualize \
--disable-wandb
```

Our training run uses 1000 iterations and completes in under 12 hours using 64 H100 GPUs.

## Acknowledgements
This codebase is built on top of the open-source implementation of [CausVid](https://github.com/tianweiy/CausVid), [Self-Forcing](https://github.com/guandeh17/Self-Forcing) and the [Wan2.1](https://github.com/Wan-Video/Wan2.1) repo.
