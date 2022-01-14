# GAME-ON

This is the source code for the paper "GAME-ON: Graph Attention Network based Multimodal Fusion for Fake News Detection."

## Directory structure

1. ```creating-graph-data``` contains the code for creating node embeddings for textual and visual graphs.

2. ```unimodal-textual```, ```unimodal-visual```, and ```multimodal-concatenation``` folders contain code for ablation study.

3. ```game-on``` contains code for the proposed framework.

4.  ```ModelSizeBaselines.ipynb``` is the code for roughly estimating the trainable paramters in the baselines. 

## Dataset

1. MediaEval 2015 (Twitter Dataset): [Link to image-verificatiion corpus github](https://github.com/MKLab-ITI/image-verification-corpus/tree/master/mediaeval2015)

2. Weibo Dataset: [Link to EANN paper for the dataset](https://github.com/yaqingwang/EANN-KDD18)


## Setup

### Dependencies

1. [dgl==0.7.0](https://github.com/dmlc/dgl/)
2. [torch==1.8.1](https://pytorch.org/get-started/locally/)
3. [torchvision==0.9.1](https://pytorch.org/get-started/locally/)
4. [transformers==4.10.2](https://huggingface.co/docs/transformers/installation)


### Run the code

1. To run any of the available models, please move to the appropriate directory

2. run ```$ python main.py ```