# Diffusion Models with Group Equivariance

Official Implementation of [Diffusion Models with Group Equivariance](https://openreview.net/forum?id=65XylEuDLB)(link to be updated). This work is a follow up to [Structure Preserving Diffusion Models](https://arxiv.org/abs/2402.19369).

<p align="center">
  <img src="assets/teaser.png" width="60%"/>
</p>

## Abstract
In recent years, diffusion models have risen to prominence as the foremost technique for distribution learning. This paper focuses on structure-preserving diffusion models (SPDM), a subset of diffusion processes tailored to distributions with inherent structures, such as group symmetries. We complement existing sufficient conditions for constructing SPDM by proving complementary necessary ones. Additionally, we propose a new framework that considers the geometric structures affecting the diffusion process. Within this framework, we propose a method of preserving the alignment between endpoint couplings in bridge models to design a novel structure-preserving bridge model. We validate our findings over a variety of equivariant diffusion models by learning symmetric distributions and the transitions among them. Empirical studies on real-world medical images indicate that our models adhere to our theoretical framework, ensuring equivariance without compromising the quality of sampled images. Furthermore, we showcase the practical utility of our framework by achieving reliable equivariant image noise reduction and style transfer, irrespective of prior knowledge of image orientation, by implementing an equivariant denoising diffusion bridge model (DDBM).

# Useage
There exists two primary branches two this repository: 'main' contains all the code for the central (pixel-space) model used for all benchmarks at low resolutions $(h\times w < 128\times 128)$, 'vqgan-model' contains the code for the latent space diffusion model based around Stable Diffusion's VAE used for high resolutions tasks $(h\times w > 128\times 128)$. See the repository [Fine-tuning Stable Diffusions VAE](https://github.com/SpencerSzabados/Fine-tune-Stable-Diffusion-VAE) for more details about how exactly the VAE was fine-tuned, and how the replicate the results yourself.

## Environment setup
We include a [Docker](https://www.docker.com/) buildfile in '/Group-Diffusion-Bridge/docker' that builds a suitable environment for running all the code by simply running the following docker command (which is also listed in '/Group-Diffusion-Bridge/docker/run_container.sh'). This docker image should download the latest version of the diffusion model code from this repository.

```sh
docker build -t group-diffusion-bridge:latest /Group-Diffusion-Bridge/docker/Dockerfile 
```
Alternatively you may use [Conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html). You can build a virtual Conda environment by running the following:
```sh
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install packaging ninja
conda install -c conda-forge mpi4py openmpi
pip install -e .
```

## General usage
### Dataset generation and loading
The /datasets/ directory contains scripts for generating/downloading various image datasets. Note, many of the datasets used require third party license agreements before they can be downloaded, hence, we are not able to directly provide methods for downloading the raw data. Additionally, due to privacy restrictions we are not able to provide any checkpoints for the PET-CT model shown in the paper.

### Training models
The provided model accepts a variety of different launch options configured within the 'args.sh' file.

The bash files [train_ddbm.sh](train_ddbm.sh) and [sample_ddbm.sh](sample_ddbm.sh) are used for model training and sampling respectively. Additionally, the bash file [train_ddbm_incremental](train_ddbm_incremental.sh) performs incremental training with configurable checkpoint and small batch sampling intervals. 

Simply set variables `DATASET_NAME` and `SCHEDULE_TYPE`:
- `DATASET_NAME` specifies which dataset to use. For each dataset, make sure to set the respective `DATA_DIR` variable in `args.sh` to your dataset path.
- `SCHEDULE_TYPE` denotes the noise schedule type. Only `ve` and `vp` are recommended. `ve_simple` and `vp_simple` are their naive baselines.

To train, run
```sh
bash train_ddbm.sh $DATASET_NAME $SCHEDULE_TYPE 
```
To resume, set CKPT to your checkpoint, or it will automatically resume from your last checkpoint based on your experiment name; e.g.,
```sh
bash train_ddbm.sh $DATASET_NAME $SCHEDULE_TYPE $CKPT
```

## Evaluations
One can evaluate samples with [evaluations/evaluator.py](evaluations/evaluator.py). 

To evaluate, set `REF_PATH` to the path of your reference stats and `SAMPLE_PATH` to your generated `.npz` path. You can additionally specify the metrics to use via `--metric`. We only support `fid` and `lpips`.

```sh
python $REF_PATH $SAMPLE_PATH --metric $YOUR_METRIC
```

## Troubleshooting
There is currently a bug within DDP when launching as parallel training task on NVIDIA A40, L40s GPUs, see [pytroch/issues](https://github.com/pytorch/pytorch/issues/73206).


# Code attribution
The given implementation(s) is initially based on the github repository of [Denoising Diffusion Bridge Models](https://github.com/alexzhou907/DDBM), components from the [EDM](https://github.com/NVlabs/edm) repository, [k-diffusion](https://github.com/crowsonkb/k-diffusion), and [Stable Diffusion](https://github.com/CompVis/stable-diffusion).

# Citation
```bibtex
@inproceedings{luSY:2024,
    author = {Lu, Haoye and Szabados, Spencer and Yu, Yaoliang},
    title = {Diffusion Models with Group Equivariance},
    booktitle = {ICML 2024 Workshop on Structured Probabilistic Inference {\&} Generative Modeling},
    year = {2024},
    url = {https://openreview.net/forum?id=65XylEuDLB}
    }
```
