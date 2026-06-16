# CRLM-GAN: a feature-constrained GAN-based deep learning framework for multi-parametric MRI-based segmentation of colorectal liver metastases before and after chemotherapy 
## News
Our paper was accepted by Cancer Imaging (JCR Q1)!
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

## Citation
```
@article{xia2025crlm,
  title={CRLM-GAN: a feature-constrained GAN-based deep learning framework for multi-parametric MRI-based segmentation of colorectal liver metastases before and after chemotherapy},
  author={Xia, Shao-Jun and Zhu, Hai-Bin and Gu, Xiao-Lei and Bao, Jing and Sun, Anlan and Cui, Yong and Li, Xiao-Ting and Sun, Ying-Shi},
  journal={Cancer Imaging},
  year={2025},
  publisher={Springer}
}
```
