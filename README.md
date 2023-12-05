# Cross-PCC
This is the official implementation for the paper [Leveraging Single-View Images for Unsupervised 3D Point Cloud Completion](https://arxiv.org/pdf/2212.00564.pdf)

## Environment
Python: 3.9  
PyTorch: 0.10.1  
Cuda: 11.1  

## Dataset
dataset url: [3DEPN](https://drive.google.com/drive/folders/1BQg9r6RT0xZ3VZzc0NswiEGGVvE3FwD4?usp=sharing)

## Get Started
### Build Extensions
```
cd pointnet2_ops_lib
python setup.py install
cd ...
cd Chamfer3D
python setup.py install
```
### Training
```
CUDA_VISIBLE_DEVICES=0 python main.py
```

## Acknowledgements
Some codes are borrowed from [SnowflakeNet](https://github.com/AllenXiangX/SnowflakeNet),[ViPC](https://github.com/Hydrogenion/ViPC) and [PCN](https://github.com/wentaoyuan/pcn). Thanks for their great works.

## Cite this work
```
@article{wu2022leveraging,
  title={Leveraging Single-View Images for Unsupervised 3D Point Cloud Completion},
  author={Wu, Lintai and Zhang, Qijian and Hou, Junhui and Xu, Yong},
  journal={arXiv preprint arXiv:2212.00564},
  year={2022}
}
```
