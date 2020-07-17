# Regression-based 3D Pose Estimation for Texture-less Objects

## Introduction

In this paper, we propose a novel network structure to estimate 3D pose for texture-less objects. The network contains two parts, the triplet network, to extract expected features from images; and the regression network, to directly regress the 3D pose using the features output by the triplet network. 

## Citation

If you find this code useful in your research, please consider citing:

    @article{liu2019regression,
        Auther = {Liu, Yuanpeng and Zhou, Laishui and Zong, Hua and Gong, Xiaoxi and Wu, Qiaoyun and Liang, Qingxiao and Wang, Jun},
        Title = {Regression-based 3D Pose Estimation for Texture-less Objects},
        Journal = {IEEE Transactions on Multimedia},
	    year={2019}
    }

## Installation

### Suggested environment
- Tensorflow >= 1.4.0
- Keras >= 2.1.0
- CUDA >= 8.0

### Instruction
1. Download [dataset](http://3dgp.net/paper/2019/attachments/dataset-3dpr.zip)

2. Unzip dataset:
```
cd ./dataset
unzip dataset-3dpr.zip
```

3. Extract features correlated to poses:
```
python ./FeatureExtraction.py
```

4. Regress poses using features:
```
python ./Regression.py
```

### Note

The matching scores are evaluated using [sixd_toolkit](https://github.com/thodan/sixd_toolkit), thanks to [Tomas Hodan](http://www.hodan.xyz).