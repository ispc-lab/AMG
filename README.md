# AMG
 The official repository of our paper "Deep reinforcement learning as an interaction agent to steer fragment-based 3D molecular generation for protein pockets". 
 
 :star2: The complete repository is under construction (updated on July 24, 2024).


## Prerequisites
- We have presented the conda environment file in `./environment.yml`.
- We have evaluated our models use external tools, including: [Qvina](https://qvina.github.io/), [Pyscreener](https://github.com/coleygroup/pyscreener).

## Dataset
- We pre-trained our model using the natural product dataset [COCONUT](https://coconut.naturalproducts.net) and the Pocket3D dataset collected from the [Protein Data Bank](https://www.rcsb.org/). The dataset used for fine-tuning was obtained from [CrossDocked2020](https://bits.csb.pitt.edu/files/crossdock2020/).
- To facilitate your implementation, we have provided the raw datasets used by AMG. Download the dataset archive from [AMG-DATA](https://drive.google.com/drive/folders/1YmqKfIDiDWkRVJGGcoPtCVPwhMPSHi2E).

## Training
Ligand encoder and fragment-based decoder pre-training:
```
python pretrain_ligand.py
```

Pocket encoder pre-training:
```
python pretrain_pocket.py
```

The first training stage:
```
python train_rec.py
```

The second training stage:
```
python train_agent.py
```

## Sampling
```
python sample.py
```
