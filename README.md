# CRLM-GAN: a multi-feature constrained generative adversarial framework for multi-parametric MRI-based segmentation of colorectal liver metastases 
## Description
CRLM-GAN introduces an innovative generative adversarial approach tailored for colorectal liver metastases (CRLM) segmentation, leveraging constrained multi-feature attention mechanisms to enhance accuracy in multi-parametric MRI images, particularly for multifocal tumors. This pioneering framework expands the application of GANs in medical imaging, advancing the understanding of their use in metastatic cancer segmentation. The developed model has the capability of well handling five different imaging conditions of CRLM, including pre-NACT DWI, pre-NACT T2WI, post-NACT DWI, post-NACT T2WI, and contrast-enhanced CT. By utilizing constrained multi-feature attention, CRLM-GAN effectively addresses the challenges of complex metastatic tumor structures. To comprehensively evaluate the performance of our proposed framework, we compare CRLM-GAN against several representative segmentation architectures, including U-Net, nnU-Net, TransUNet, SegAN, and NestedUNet. The primary objective of this project is to develop a deep learning-based tool to assist abdominal radiologists in accurately segmenting CRLM. Rigorous evaluation demonstrates the framework’s robust capability to handle multi-parametric MRI data, offering a reliable solution for metastatic tumor segmentation.
## Architecture of CRLM-GAN model 
This model utilizes a UNet++ network as the generator and a pre-trained ResNet50 as the discriminator, incorporating multi-scale feature extraction. The network is trained by optimizing binary cross-entropy loss, Dice loss, and multi-feature constrained loss to enhance segmentation performance.

<img width="4794" height="2860" alt="图片 1" src="https://github.com/user-attachments/assets/45432007-9128-4a9c-8e0e-74ecfc89e9e3" />


## Segmentation samples on the multi-parametric MRI dataset
Segmentation samples on the multi-parametric MRI dataset, compared with ROIs from radiologists.

<img src="https://github.com/Xuezai-wq/CRLM-GAN/blob/main/figure3.png">

## Segmentation samples on the CT dataset
Visualization results of segmentation in CT images, with CRLM tumors ranging from 1 to 4.

<img src="https://github.com/Xuezai-wq/CRLM-GAN/blob/main/figure4.png">

## Citations
```
@article{Isensee2021nnUNet,
  title = {nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author = {Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and Petersen, Jens and Maier-Hein, Klaus H.},
  journal = {Nature Methods},
  volume = {18},
  number = {2},
  pages = {203--211},
  year = {2021},
  doi = {10.1038/s41592-020-01008-z}
}

@article{Chen2021TransUNet,
  title = {TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author = {Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and … Yuille, Alan L.},
  journal = {arXiv preprint arXiv:2102.04306},
  year = {2021}
}

@article{Xue2018SegAN,
  title = {SegAN: Adversarial Network with Multi-scale L₁ Loss for Medical Image Segmentation},
  author = {Xue, Yuan and Xu, Tao and Zhang, Han and Long, L. Rodney and Huang, Xiaolei},
  journal = {Neuroinformatics},
  volume = {16},
  number = {3-4},
  pages = {383--392},
  year = {2018},
  doi = {10.1007/s12021-018-9377-x}
}

@article{Zhou2018UNetpp,
  title = {UNet++: A Nested U-Net Architecture for Medical Image Segmentation},
  author = {Zhou, Zongwei and Siddiquee, Md Mahfuzur Rahman and Tajbakhsh, Nima and Liang, Jianming},
  journal = {arXiv preprint arXiv:1807.10165},
  year = {2018}
}
@inproceedings{Ronneberger2015UNet,
  title     = {U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author    = {Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015},
  series    = {Lecture Notes in Computer Science},
  volume    = {9351},
  pages     = {234--241},
  publisher = {Springer, Cham},
  year      = {2015},
  doi       = {10.1007/978-3-319-24574-4_28}
}
```
