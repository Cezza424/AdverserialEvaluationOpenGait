# Project artefact how-to guide



This repository contains the artefact for the research and development project: 'Empirically evaluating the robustness of multimodal gait recognition systems'. It was built using the OpenGait framework.



To reproduce the experiment carried out in this project follow theses steps:

### Setup

Once the repo has been downloaded, install these dependencies:

* pytorch cuda
* torchvision
* pyyaml
* tensorboard
* opencv-python
* tqdm
* py7zr
* kornia
* einops

### Dataset



The pre-processed dataset that is used in the experiment can be accessed via [OneDrive](https://edgehill-my.sharepoint.com/:u:/g/personal/25299867_edgehill_ac_uk/IQCpcS0oRHPaSKeeWFmv-f1QAd8L_HXTxekevUpSqkRf6jk?e=s0ERiW). Download the zip file and extract it.

The original dataset, before it was pre-processed to be used with OpenGait in the experiment, is [Health\&Gait](https://zenodo.org/records/14039922)

### Configuration

* In the configuration file '[skeletongait++\_CCPG\_Fine-tunedHealthGait.yaml](https://github.com/Cezza424/AdverserialEvaluationOpenGait/blob/a1e76121c0f6b386ea99e5080660f402d662b174/configs/skeletongait/skeletongait%252B%252B_CCPG_Fine-tunedHealthGait.yaml)' set the dataset\_root field to the location of the dataset downloaded from OneDrive and use the absolute path.
* In the dataset\_partition field, enter the location of the partition file '[opengait\_partition\_3.json](https://github.com/Cezza424/AdverserialEvaluationOpenGait/blob/a1e76121c0f6b386ea99e5080660f402d662b174/datasets/opengait_partition_3.json)' and use the absolute path.

#### Model weights checkpoint

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffc107)](https://huggingface.co/Cezza424/SkeletonGaitPP_HealthGait_Finetune)

Download the model's weights and put the file in the '[checkpoints](https://github.com/Cezza424/AdverserialEvaluationOpenGait/tree/ea00c3fd1b096195dfcd6e8fe52874c2746cd0c4/output/HealthGait/SkeletonGaitPP/SkeletonGaitPP_HealthGait_Finetune/checkpoints)' folder

#### [Pipeline](https://github.com/Cezza424/AdverserialEvaluationOpenGait/blob/ea00c3fd1b096195dfcd6e8fe52874c2746cd0c4/opengait/run_adversarial_eval.py)

To reproduce the experiment, ensure that the modes list (on [ln 47](https://github.com/Cezza424/AdverserialEvaluationOpenGait/blob/ac595c92bab3c9ec886cc322173eab95ecbe838c/opengait/run_adversarial_eval.py#L47)) in the [Pipeline](https://github.com/Cezza424/AdverserialEvaluationOpenGait/blob/ea00c3fd1b096195dfcd6e8fe52874c2746cd0c4/opengait/run_adversarial_eval.py) file called 'run\_adversarial\_eval.py' has each of these modes 'baseline, attack\_silhouette, attack\_skeleton, combined\_attack'.

Then run

```
python opengait/run_adversarial_eval.py --cfgs C:\Users\25299867\PycharmProjects\OpenGait\configs\deepgaitv2\DeepGaitV2_ccpg.yaml --log_to_file --single_run
```

remove the --single\_run flag for the iterative attacks mode and set the [epsilon](https://github.com/Cezza424/AdverserialEvaluationOpenGait/blob/ac595c92bab3c9ec886cc322173eab95ecbe838c/configs/skeletongait/skeletongait%252B%252B_CCPG_Fine-tunedHealthGait.yaml#L114) and [flip\_prob](https://github.com/Cezza424/AdverserialEvaluationOpenGait/blob/ac595c92bab3c9ec886cc322173eab95ecbe838c/configs/skeletongait/skeletongait%252B%252B_CCPG_Fine-tunedHealthGait.yaml#L117) values in the [config file](https://github.com/Cezza424/AdverserialEvaluationOpenGait/blob/a1e76121c0f6b386ea99e5080660f402d662b174/configs/skeletongait/skeletongait%252B%252B_CCPG_Fine-tunedHealthGait.yaml) to 0.

