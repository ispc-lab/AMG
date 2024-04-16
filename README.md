# AMG
 The official repository of our paper "Deep reinforcement learning as an interaction agent to steer fragment-based 3D molecular generation for protein pockets"


## Prerequisites
- python3, pytorch, numpy, etc.
- We have presented the our conda environment file in `./environment.yml`.

## Dataset
- We pre-trained our model using the natural product dataset [COCONUT](https://coconut.naturalproducts.net) and the Pocket3D dataset collected from the [Protein Data Bank](https://www.rcsb.org/). The dataset used for fine-tuning was obtained from [CrossDocked2020](https://bits.csb.pitt.edu/files/crossdock2020/).
- To facilitate your implementation, we have provided the raw datasets used by AMG. Download the dataset archive from [AMG-DATA](https://drive.google.com/drive/folders/1YmqKfIDiDWkRVJGGcoPtCVPwhMPSHi2E).

## Training
Ligand encoder and fragment-based pre-training:
```
python pretrain_ligand.py
```

Pocket encoder pre-training:
```
python pretrain_pocket.py
```

The first training stage:
```
python train_first.py
```

The second training stage:
```
python train_second.py
```

## Sampling
```
python sample.py
```
