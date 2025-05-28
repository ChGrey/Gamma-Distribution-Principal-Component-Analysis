# Gamma-Distribution Principal Component Analysis
## Abstract
Scattering characteristics of synthetic aperture radar (SAR) targets are typically related to observed azimuth and depression angles. However, in practice, it is difficult to obtain adequate training samples at all observation angles, which probably leads to poor robustness of deep networks. In this paper, we first propose a Gamma-Distribution Principal Component Analysis (ΓPCA) model that fully accounts for the statistical characteristics of SAR data. The ΓPCA derives consistent convolution kernels to effectively capture the angle-invariant features of the same target at various attitude angles, thus alleviating deep models' sensitivity to angle changes in SAR target recognition task. We validate ΓPCA model based on two commonly used backbones, ResNet and ViT, and conducted multiple robustness experiments on the MSTAR benchmark dataset. The experimental results demonstrated that ΓPCA effectively enables the model to withstand substantial distributional discrepancy caused by angle changes. Additionally, ΓPCA convolution kernel is designed to require no parameter updates, thereby bringing no computational burden to a network.
## Introduction
Implementation of Gamma-Distribution Principal Analysis (ΓPCA).

<img src="https://github.com/ChGrey/Gamma-Distribution-Principal-Component-Analysis/raw/main/img//gpca_model.png" alt="img1">

ΓPCA feature illustration.

<img src="https://github.com/ChGrey/Gamma-Distribution-Principal-Component-Analysis/raw/main/img/conv_output.png" alt="img2" width="50%">
