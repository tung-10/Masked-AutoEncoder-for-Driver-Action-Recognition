# MAE4DAR: Masked AutoEncoder for Driver Action Recognition

This repository contains the implementation of the research presented in the paper **"MAE4DAR: Masked AutoEncoder for Driver Action Recognition"**. 

## 1. Dataset Preparation

We evaluate our method on the **UTCDriverAct** dataset. 

## 2. Installation & Requirements

The required packages are in the file `requirements.txt`, and you can run the following command to install the environment

```
conda create --name videomae python=3.8 -y
conda activate videomae

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch

pip install -r requirements.txt
```

### Note:
- **The above commands are for reference only**, please configure your own environment according to your needs.
- We recommend installing **`PyTorch >= 1.12.0`**, which may greatly reduce the GPU memory usage.
- It is recommended to install **`timm == 0.4.12`**, because some of the APIs we use are deprecated in the latest version of timm.

## 3. Model Finetuning

Model is initialized with pretrained VideoMAE with Kinetic-700. Run this command to train the model on segmented videos dataset:

```
bash scripts/train_UTCDA.sh
```

## 4. Validation Pipeline

The evaluating process is split into Model inference stage and the Post-processing stage to obtain the final recognition results.

### Step 4.1: Model Inference (`evaluate_loc.py`)
Use the script `eval_UTCDA.sh` that runs `evaluate_loc.py` to review how well the model recognizes behaviors on the sampling scenario. 

```
bash scripts/eval_UTCDA.sh
```
After completing the process, Top1 Accuracy is printed into the screen.

### Step 4.2: Post-processing (`postprocessing.py`)
Use `postprocessing.py` to implement the post-processing and generate the final action localization results with Temporal overlap score.

```
python postprocessing.py
```
To get Frame-wise Accuracy result, run the following command and check paths if need be:
```
python frame_wise_acc.py
```
### Step 4.3: Multi-view Fusion (obtional)
After obtaining model inferences for different views, run the following command to generate the late fusion results with post-processing:
```
python post_combine.py
```
## Citation