# know_your_mind
## Title: Know Your Mind: Adaptive Brain Signal Classification with Reinforced Attentive Convolutional Neural Networks

**PDF: [ICDM 2019](https://www.computer.org/csdl/proceedings-article/icdm/2019/460400a896/1h5XNhDJV1S), [arXiv](https://arxiv.org/abs/1802.03996)**

**Authors: [Xiang Zhang](http://xiangzhang.info/) (xiang_zhang@hms.harvard.edu), [Lina Yao](https://www.linayao.com/) (lina.yao@unsw.edu.au), Xianzhi Wang, Wenjie Zhang, Shuai Zhang, Yunhao Liu**

## Overview
This repository contains reproducible codes for the proposed model.
This paper proposes a generic and effective framework for raw EEG signal classification to support the development of BCI applications. The framework works directly on raw EEG data without requiring any preprocessing or feature engineering. Besides, it can automatically select distinguishable featuredimensions for different EEG data, thus achieving high usability.The experimental results demonstrate that our approach not only outperforms several state-of-the-art baselines by a large margin but also shows low latency and high resilience in coping with multiple EEG signal channels and incomplete EEG signals. Our approach applies to wider application scenarios such as intention recognition, person identification, and neurological diagnosis.


## Code
1. The Dueling DQN codes including: 
  - Environment code: `env_spatial_autoregressive.py`, which is the most valuable part. Please design your environment based on your classification task. It can not only used on EEG data classification but can be used for any policy optimization.
  - Agent: `RL_brain.py`. 
  - Main: `run_this.py`


2. In this paper we regard the combination of the convolutional mapping procedure and the classifier as the reward model. To make it more readable and help to design your own reward model, we separately write the reward model as a single python file: `B_1DCNN.py`. Please design your own reward model corresponding to your problem and insert it in the `env_spatial_autoregressive.py`. I left the insert position blank and added comments already.


## Citing
If you find our work useful for your research, please consider citing this paper:

    @article{zhang2018know,
      title={Know your mind: adaptive brain signal classification with reinforced attentive convolutional neural networks},
      author={Zhang, Xiang and Yao, Lina and Wang, Xianzhi and Zhang, Wenjie and Zhang, Shuai and Liu, Yunhao},
      journal={arXiv preprint arXiv:1802.03996},
      year={2018}
    }

## Datasets
- The eegmmidb dataset can be downloaded in this link:
https://www.physionet.org/pn4/eegmmidb/

- The emotiv is our local dataset, please refer to our PerCom 2018 paper:Converting Your Thoughts to Texts: Enabling Brain Typing via Deep Feature Learning of EEG Signals. The download link: 
https://drive.google.com/open?id=0B9MuJb6Xx2PIM0otakxuVHpkWkk

- The EEG-S is a subset of the eegmmidb dataset. Take one single task and mark the person ID as label. Classifying the person ID, which is so called person identification.

- The TUH is an excellent public EEG dataset released by Temple University Hospotial. Download here: https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml  
In this paper, we are using the TUH EEG Seizure Corpus (v1.0.3) released on 26/04/2017. This dataset is 11.6 G, which is too big that we only use a subset of it in our paper.



## Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <xiang.alan.zhang@gmail.com>.


## License

This repository is licensed under the MIT License.
