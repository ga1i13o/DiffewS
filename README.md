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
<!-- - [ ] Release the weights. -->
- [x] Release the weights.
- [x] Release the inference code.
- [x] Release the training code.
<!-- --- -->

## 👻 Getting Started

### Installation
Preparing the environment following [GenPercept](https://github.com/aim-uofa/GenPercept?tab=readme-ov-file).


```bash
conda create -n diffews python=3.10
conda activate diffews
pip install -r requirements.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Dataset
Preparing the dataset following [Matcher](https://github.com/aim-uofa/Matcher/blob/main/datasets/README.md)
You only need to download the COCO 2014 dataset.

### Training

This script is tested on single 24G 4090.
```bash
bash scripts/train_cocofold0_4090_nocrop_lr1_nearest_fold1_7shot_ori_v3.sh
```
### Evaluation

Download the pre-trained model weights from [here](https://www.modelscope.cn/zzzmmz/Diffews.git). 
```bash
CUDA_VISIBLE_DEVICES=0 bash  scripts/eval_coco2014_rthres_1shot_nosample.sh weight/coco_fold0
CUDA_VISIBLE_DEVICES=0 bash  scripts/eval_coco2014_rthres_5shot_nosample.sh weight/coco_fold0
CUDA_VISIBLE_DEVICES=0 bash  scripts/eval_coco2014_rthres_1shot_nosample_fold0.sh weight/incontext
```

## 🎫 License

For academic use, this project is licensed under [the 2-clause BSD License](LICENSE). For commercial use, please contact [Chunhua Shen](mailto:chhshen@gmail.com).

## 🖊️ Citation


If you find this project useful in your research, please consider to cite:


```BibTeX
@article{zhu2024unleashing,
  title={Unleashing the Potential of the Diffusion Model in Few-shot Semantic Segmentation},
  author={Zhu, Muzhi and Liu, Yang and Luo, Zekai and Jing, Chenchen and Chen, Hao and Xu, Guangkai and Wang, Xinlong and Shen, Chunhua},
  journal={arXiv preprint arXiv:2410.02369},
  year={2024}
}
```

## Acknowledgement
[SegGPT](https://github.com/baaivision/Painter/tree/main/SegGPT), [Matcher](https://github.com/aim-uofa/Matcher), [Marigold](https://github.com/prs-eth/Marigold), [GenPercept](https://github.com/aim-uofa/GenPercept?tab=readme-ov-file)
