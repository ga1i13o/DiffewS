<div align="center">

<h1>A Simple Image Segmentation Framework via In-Context Examples </h1>

[Yang Liu](https://scholar.google.com/citations?user=9JcQ2hwAAAAJ&hl=en)<sup>1</sup>, &nbsp; 
[Chenchen Jing](https://jingchenchen.github.io/)<sup>1</sup>, &nbsp;
Hengtao Li<sup>1</sup>, &nbsp;
[Muzhi Zhu](https://scholar.google.com/citations?user=064gBH4AAAAJ&hl=en)<sup>1</sup>, &nbsp;
[Hao Chen](https://stan-haochen.github.io/)<sup>1</sup>, &nbsp;
[Xinlong Wang](https://www.xloong.wang/)<sup>2</sup>, &nbsp;
[Chunhua Shen](https://cshen.github.io/)<sup>1</sup>

<sup>1</sup>[Zhejiang University](https://www.zju.edu.cn/english/), &nbsp;
<sup>2</sup>[Beijing Academy of Artificial Intelligence](https://www.baai.ac.cn/english.html)

NeurIPS 2024

</div>

## 🚀 Overview
<div align="center">
<img width="800" alt="image" src="figs/framework.png">
</div>

## 📖 Description

Recently, there have been explorations of generalist segmentation models that can effectively tackle a variety of image segmentation tasks within a unified in-context learning framework. 
However, these methods still struggle with task ambiguity in in-context segmentation, as not all in-context examples can accurately convey the task information. 
In order to address this issue, we present SINE, a simple image **S**egmentation framework utilizing **in**-context **e**xamples. 
Our approach leverages a Transformer encoder-decoder structure, where the encoder provides high-quality image representations, and the decoder is designed to yield multiple task-specific output masks to effectively eliminate task ambiguity.

[Paper](https://arxiv.org/abs/2410.04842)


## 👻 Getting Started

- [Training](TRAINING.md). 

- DINOv2 model trained on ADE20K, COCO, and Objects365, [weight](https://drive.google.com/file/d/1GYQbbUZClbmhVESDLpRwqe-TyijW2kKb/view?usp=sharing).

- [Evaluation - Few-shot Semnatic Segmentation](inference_fss/EVALUATION.md)

- [Evaluation - Few-shot Instance Segmentation](inference_fsod/EVALUATION.md)

- [Evaluation - Video Object Segmentation](inference_vos/EVALUATION.md)



## 🎫 License

For academic use, this project is licensed under [the 2-clause BSD License](LICENSE). For commercial use, please contact [Chunhua Shen](chhshen@gmail.com).

## 🖊️ Citation


If you find this project useful in your research, please consider cite:


```BibTeX
@article{liu2024simple,
  title={A Simple Image Segmentation Framework via In-Context Examples},
  author={Liu, Yang and Jing, Chenchen and Li, Hengtao and Zhu, Muzhi and Chen, Hao and Wang, Xinlong and Shen, Chunhua},
  journal={arXiv preprint arXiv:2410.04842},
  year={2024}
}
```

## Acknowledgement
[DINOv2](https://github.com/facebookresearch/dinov2), [Mask2Former](https://github.com/facebookresearch/Mask2Former), [SegGPT](https://github.com/baaivision/Painter/tree/main/SegGPT), [Matcher](https://github.com/aim-uofa/Matcher), [TFA](https://github.com/ucbdrive/few-shot-object-detection) and [detectron2](https://github.com/facebookresearch/detectron2).
