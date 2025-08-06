# Robust deformable image registration using synthetic data and transfer learning
## Iris D. Kolenbrander, Matteo Maspero, Josien P.W. Pluim  

This repository contains code for pre-training and fine-tuning deep learning-based registration models.
It also provides the download links to the pre-trained model, which can be used for fine-tuning on target datasets. 
Paper is available [here](https://www.melba-journal.org/private/5dab7a968f8be7181d424adbdfc90478dcaab540d317c368c3908e466cc6fb9d.html).

## Citation
If you use this repository or its contents in your research, please cite the following paper:
> Kolenbrander, I. D., Maspero, M., & Pluim, J. P. W. (2025). Robust deformable image registration using synthetic data and transfer learning. Machine Learning for Biomedical Imaging, 3(August 2025 issue), 287–316.
---

## Models
The trained model weights are also in the repository, under /model_weights.

---
## Code

We used **PyTorch Lightning** for training and inference.

To help users test the pipeline, we provide **toy data** (only 2 cases).  
The code is structured into two main stages:

```bash
python -m configs.stage_01_pretrain_synthetic_data


### 1. Pre-training on synthetic data
python -m configs.stage_01_pretrain_synthetic_data

### 2. Training on target datasets (e.g., lung CT)
#### 2.1. From-scratch training
python -m configs.stage_02_finetune_lung_CT_data

#### 2.2. Fine-tuning (after pre-training on synthetic data)
python -m configs.stage_02_finetune_lung_CT_data --load_model pretrained_model
```
