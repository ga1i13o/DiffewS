<div align="center">

<h1>Unleashing the Potential of the Diffusion Model in Few-shot Semantic Segmentation</h1>

[Muzhi Zhu](https://scholar.google.com/citations?user=064gBH4AAAAJ&hl=en)<sup>1*</sup>, &nbsp;
[Yang Liu](https://scholar.google.com/citations?user=9JcQ2hwAAAAJ&hl=en)<sup>1*</sup>, &nbsp;
Zekai Luo<sup>1*</sup>, &nbsp; 
[Chenchen Jing](https://jingchenchen.github.io/)<sup>1</sup>, &nbsp;
[Hao Chen](https://stan-haochen.github.io/)<sup>1</sup>, &nbsp;
[Guangkai Xu](https://scholar.google.com.hk/citations?user=v35sbGEAAAAJ&hl=en)<sup>1</sup>, &nbsp;
[Xinlong Wang](https://www.xloong.wang/)<sup>2</sup>, &nbsp;
[Chunhua Shen](https://cshen.github.io/)<sup>1</sup>

<sup>1</sup>[Zhejiang University](https://www.zju.edu.cn/english/), &nbsp;
<sup>2</sup>[Beijing Academy of Artificial Intelligence](https://www.baai.ac.cn/english.html)

NeurIPS 2024

</div>

## 🚀 Overview
<div align="center">
<img width="800" alt="image" src="figs/method.png">
</div>

## 📖 Description

We systematically study four crucial elements of applying the Diffusion Model to Few-shot
Semantic Segmentation. For each of these aspects, we propose several reasonable solutions
and validate them through comprehensive experiments.

Building upon our observations, we establish the DiffewS framework, which maximally
retains the generative framework and effectively utilizes the pre-training prior. Notably, we
introduce the first diffusion-based model dedicated to Few-shot Semantic Segmentation,
setting the groundwork for a diffusion-based generalist segmentation model.

[Paper](https://arxiv.org/abs/2410.02369)

## 🚩 Plan
- [x] Release the weights.
- [x] Release the inference code.
- [ ] Release the training code.
<!-- --- -->

## 👻 Getting Started

- [Training](TRAINING.md). 

- DINOv2-L model trained on ADE20K, COCO, and Objects365, [weight](https://drive.google.com/file/d/1GYQbbUZClbmhVESDLpRwqe-TyijW2kKb/view?usp=sharing).

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
