# Fashion-U-Want: High Resolution Virtual Try-On

PyTorch implementation of Fashion U Want: High Resolution Virtual Try-On with GAN models

### Tech Stack

<p align="left">
  <img src="https://img.shields.io/badge/Library-Detectron2-blue" alt="Detectron2"/>
  <img src="https://img.shields.io/badge/Library-EfficientNet%20B7-teal" alt="EfficientNet B7"/>
  <img src="https://img.shields.io/badge/Library-U2Net-yellow" alt="U2Net"/>
  <img src="https://img.shields.io/badge/Framework-OpenPose-purple" alt="OpenPose"/>
  <img src="https://img.shields.io/badge/Framework-Graphonomy-green" alt="Graphonomy"/>
  <img src="https://img.shields.io/badge/Framework-DeepLabV3%2B-red" alt="DeepLabV3+"/>
  <img src="https://img.shields.io/badge/Framework-HR--VITON-orange" alt="HR-VITON"/>
  <img src="https://img.shields.io/badge/Model-GAN-lightgrey" alt="GAN"/>
</p>


---
Fashion-U-Want is a high-resolution virtual try-on system that uses deep learning to overlay clothing onto a person's image. This framework processes input images through ``Try_TRYON.ipynb``, extracting clothing masks, pose information, and segmentation data to generate realistic try-on results—all without requiring manual installation of dependencies.

## Project Duration

**2024.11.13 - 2024.11.20**

<table>
  <tbody>
    <tr>
      <td align="center">
        <a href="https://github.com/PARKYUNSU">
          <img src="https://github.com/PARKYUNSU.png" width="100px;" alt=""/>
          <br /><sub><b>Yunsu Park</b></sub>
        </a>
        <br />
      </td>
      <td align="center">
        <a href="https://github.com/navi0728">
          <img src="https://github.com/navi0728.png" width="100px;" alt=""/>
          <br /><sub><b>Minju Lee</b></sub>
        </a>
        <br />
      </td>
      <td align="center">
        <a href="https://github.com/MyoungJinSon">
          <img src="https://github.com/MyoungJinSon.png" width="100px;" alt=""/>
          <br /><sub><b>Myoungjin Son</b></sub>
        </a>
        <br />
      </td>
    </tr>
  </tbody>
</table>


## How to Use

1. Click the "Open in Colab" button below.
2. Follow the instructions in the notebook to upload input images and generate try-on result.
3. View the output directly within the notebook.

## Fashion-U-Want: Demo

Click the badge below to run the demo:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PARKYUNSU/Fashion-U-Want-Virtual-Try-On/blob/main/Try_TRYON.ipynb)


## Results

<img src="https://github.com/user-attachments/assets/7fe47f38-01fa-492f-b75d-972594426bd2" width=700>



<img src="https://github.com/user-attachments/assets/4d3f24c8-94d8-4777-8fdb-ca2079be035d" width=290>

## References

### Pose Estimation
- **OpenPose**: Realtime multi-person 2D pose estimation using part affinity fields.  
  Repository: [https://github.com/CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

### Semantic Segmentation
- **DeepLabV3+**: Encoder-decoder with atrous separable convolution for semantic image segmentation.  
  Repository: [https://github.com/tensorflow/models/tree/master/research/deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab)

- **U2Net**: U-Net based architecture for salient object detection and segmentation.  
  Repository: [https://github.com/xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)

- **EfficientNet B7**: High-performance image segmentation model, pre-trained on ImageNet.  
  Repository: [https://huggingface.co/google/efficientnet-b7](https://huggingface.co/google/efficientnet-b7)

### Dense Pose
- **Detectron2**: Object detection and human parsing framework for research and production.  
  Repository: [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)

### Human Parsing
- **Graphonomy**: Universal human parsing framework for garment understanding.  
  Repository: [https://github.com/Gaoyiminggithub/Graphonomy](https://github.com/Gaoyiminggithub/Graphonomy)

### Virtual Try-On
- **HR-VITON**: High-resolution virtual try-on for clothing.  
  Repository: [https://github.com/sangyun884/HR-VITON](https://github.com/sangyun884/HR-VITON)
