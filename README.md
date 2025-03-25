# CRLM-GAN: a multi-feature constrained generative adversarial framework for multi-parametric MRI-based segmentation of colorectal liver metastases 
## Abstract
CRLM-GAN introduces an innovative generative adversarial approach tailored for colorectal liver metastases (CRLM) segmentation, leveraging constrained multi-feature attention mechanisms to enhance accuracy in multi-parametric MRI images, particularly for multifocal tumors. This pioneering framework expands the application of GANs in medical imaging, advancing the understanding of their use in metastatic cancer segmentation. By utilizing constrained multi-feature attention, CRLM-GAN effectively addresses the challenges of complex metastatic tumor structures. The primary objective of this project is to develop a deep learning-based tool to assist abdominal radiologists in accurately segmenting CRLM. Rigorous evaluation demonstrates the framework’s robust capability to handle multi-parametric MRI data, offering a reliable solution for metastatic tumor segmentation.
## Architecture of CRLM-GAN model 
This model utilizes a UNet++ network as the generator and a pre-trained ResNet50 as the discriminator, incorporating multi-scale feature extraction. The network is trained by optimizing binary cross-entropy loss, Dice loss, and multi-feature constrained loss to enhance segmentation performance.
## Segmentation samples on the multi-parametric MRI dataset

## Segmentation samples on the CT dataset
This figure illustrates the segmentation results of CRLM-GAN on colorectal liver metastases(CRLM) images.

<img width="387" alt="image" src="https://github.com/user-attachments/assets/b1774516-f71e-4e4c-ae39-dc6968cc8e2e" />
