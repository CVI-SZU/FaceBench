<div align="center">  
<h2>FaceBench: A Multi-View Multi-Level Facial Attribute VQA Dataset for Benchmarking Face Perception MLLMs [CVPR 2025]</h2>
Xiaoqin Wang, Xusen Ma, Xianxu Hou, Meidan Ding, Yudong Li, Junliang Chen, Wenting Chen, Xiaoyang Peng, Linlin Shen*

[![ArXiv](https://img.shields.io/badge/ArXiv-2503.21457-B31B1B.svg)](https://arxiv.org/pdf/2503.21457)
[![Webpage](https://img.shields.io/badge/Webpage-FaceBench-<COLOR>.svg)](https://github.com/CVI-SZU/FaceBench/tree/main)
[![Dataset](https://img.shields.io/badge/HuggingFace🤗-Dataset-blue)](https://github.com/CVI-SZU/FaceBench/tree/main)
[![Models](https://img.shields.io/badge/HuggingFace🤗-Models-blue)](https://huggingface.co/wxqlab/face-llava-v1.5-13b)  
</div>

## Overview
In this work, we introduce **FaceBench**, a dataset featuring hierarchical multi-view and multi-level attributes specifically designed to assess the comprehensive face perception abilities of MLLMs. We construct a hierarchical facial attribute structure, which encompasses five views with up to three levels of attributes, totaling over **210** attributes and **700** attribute values. Based on the structure, the proposed FaceBench consists of **49,919 visual question-answering (VQA) pairs** for evaluation and **23,841 pairs for fine-tuning**. Moreover, we further develop a robust face perception MLLM baseline, **Face-LLaVA**, by training with our proposed face VQA data.
<div align="center"><img src="./assets/overview.png" width="100%" height="100%"></div>

## News
* **[2024-08-20]** The Face-LLaVA model is released on [HuggingFace](https://huggingface.co/wxqlab/face-llava-v1.5-13b)🤗.
* **[2024-03-27]** The paper is released on [ArXiv](https://arxiv.org/pdf/2503.21457)🔥.

## TODO
- [X] Release the Face-LLaVA model.
- [ ] Release the evaluation code.
- [ ] Release the dataset.

## Dataset Statistics
**Distribution of visual question-answer pairs.**
<div align="center"><img src="./assets/VQAs.jpg" width="100%" height="100%"></div>

**Some samples from our dataset.**
<div align="center"><img src="./assets/example.png" width="100%" height="100%"></div>

## Evaluation

## Results
**Experimental results of various MLLMs and our Face-LLaVA across five facial attribute views.**
<div align="center"><img src="./assets/five-view-results.jpg" width="100%" height="100%"></div>

**Experimental results of various MLLMs and our Face-LLaVA across Level 1 facial attributes.**
<div align="center"><img src="./assets/level-1-results.jpg" width="100%" height="100%"></div>

## Citation
If you find this work useful for your research, please consider citing our paper:
```
@inproceedings{wang2025facebench,
  title={FaceBench: A Multi-View Multi-Level Facial Attribute VQA Dataset for Benchmarking Face Perception MLLMs},
  author={Wang, Xiaoqin and Ma, Xusen and Hou, Xianxu and Ding, Meidan and Li, Yudong and Chen, Junliang and Chen, Wenting and Peng, Xiaoyang and Shen, Linlin},
  booktitle={Proceedings-2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025},
  year={2025}
}

@article{wang2025facebench,
  title={FaceBench: A Multi-View Multi-Level Facial Attribute VQA Dataset for Benchmarking Face Perception MLLMs},
  author={Wang, Xiaoqin and Ma, Xusen and Hou, Xianxu and Ding, Meidan and Li, Yudong and Chen, Junliang and Chen, Wenting and Peng, Xiaoyang and Shen, Linlin},
  journal={arXiv preprint arXiv:2503.21457},
  year={2025}
}
```
If you have any questions, you can either create issues or contact me by email wangxiaoqin2022@email.szu.edu.cn.

## Acknowledgments
This work is heavily based on [LLaVA](https://github.com/haotian-liu/LLaVA). Thanks to the authors for their great work.
