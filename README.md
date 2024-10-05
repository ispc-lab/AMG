# AMG
:boom: The official repository of our paper "Deep reinforcement learning as an interaction agent to steer fragment-based 3D molecular generation for protein pockets". 

## Introduction
Designing high-affinity molecules for protein targets (especially novel protein families) is a crucial yet challenging task in drug discovery. Recently, there has been tremendous progress in structure-based 3D molecular generative models that incorporate structural information of protein pockets. However, the capacity for molecular representation learning and the generalization for capturing interaction patterns need substantial further developments. Here, we propose AMG, a framework that leverages deep reinforcement learning as a pocket-ligand interaction agent to gradually steer fragment-based 3D molecular generation targeting protein pockets. AMG is trained using a two-stage strategy to capture interaction features and explicitly optimize the interaction agent. The framework also introduces a pair of separate encoders for pockets and ligands, coupled with a dedicated pre-training strategy. This enables AMG to enhance its generalization ability by leveraging a vast repository of undocked pockets and molecules, thus mitigating the constraints posed by the limited quantity and quality of available datasets. Extensive evaluations demonstrate that AMG significantly outperforms five state-of-the-art baselines in affinity performance while maintaining proper drug-likeness properties. Furthermore, visual analysis confirms the superiority of AMG at capturing 3D molecular geometrical features and interaction patterns within pocket-ligand complexes, indicating its considerable promise for various structure-based downstream tasks.

## Framework
<p align="center">
  <img width="700" src="assets/overview.png" /> 
</p>

## Prerequisites
We have presented the conda environment file in `./environment.yml`.

We have evaluated our models using external tools, including: [Qvina](https://qvina.github.io/), [Pyscreener](https://github.com/coleygroup/pyscreener).

## Install via Conda and Pip
```
conda create -n AMG python=3.7
conda activate AMG
conda install pytorch==1.13.1  pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge pdbfixer
conda install conda-forge::openbabel

pip install tensorboard==1.15.0
pip isntall protobuf==3.19.6
pip install networkx==2.6.3
pip install rdkit==2023.3.2
pip install biopython==1.81
pip install pyscreener==1.1.1
pip install -U "ray[default]"

cd ADFRsuite_x86_64Linux_1.0
./install.sh -d myFolder -c 0

cd spinningup
pip install -e .
```


## Dataset
:star2: We pre-trained our model using the natural product dataset [COCONUT](https://coconut.naturalproducts.net) and the Pocket3D dataset collected from the [Protein Data Bank](https://www.rcsb.org/). The dataset used for fine-tuning was obtained from [CrossDocked2020](https://bits.csb.pitt.edu/files/crossdock2020/).

 :star2: To facilitate your implementation, we have provided the raw datasets used by AMG. Download the dataset archive from [AMG-DATA](https://drive.google.com/drive/folders/1YmqKfIDiDWkRVJGGcoPtCVPwhMPSHi2E).

## Training
Ligand encoder and fragment-based decoder pre-training:
```
python scripts/pretrain_ligand.py  
```

Pocket encoder pre-training:
```
python scripts/pretrain_pocket.py
```

The first training stage:
```
python scripts/train_rec.py
```

The second training stage:
```
python scripts/train_agent.py
```

## Sampling
```
python scripts/sample_testset.py --config configs/rl.yml --start_index 0  --end_index 99 
```

## Evaluation
### Evaluation from sampling results
```
python scripts/evaluate_amg.py
```

### Evaluation from meta files
We provide the sampling results of our model and Pocket2Mol, TargetDiff, DecompDiff, ResGen, FLAG baselines [here](https://drive.google.com/drive/folders/1Gk4WFUiIQsp2UKT9dMI4OVljbIzzDZJs).

You can directly reproduce the results reported in the paper quickly with [summary.ipynb](https://github.com/ispc-lab/AMG/blob/main/notebooks/summary.ipynb).
