# MedContext:  Learning Contextual Cues for Efficient Volumetric Medical Segmentation [MICCAI'24]
(full code and all the models will be released soon!)

[Hanan Gani<sup>1</sup>](https://hananshafi.github.io/), [Muzammal Naseer<sup>1</sup>](https://muzammal-naseer.com/), [Fahad Khan<sup>1,2</sup>](https://sites.google.com/view/fahadkhans/home), [Salman Khan<sup>1,3</sup>](https://salman-h-khan.github.io/)

<sup>1</sup>Mohamed Bin Zayed University of AI      <sup>2</sup>Linkoping University      <sup>3</sup>Australian National University

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2402.17725)

Official code for the paper "MedContext: Learning Contextual Cues for Efficient Volumetric Medical Segmentation".

<hr>

## Contents

1. [Updates](#News)
2. [Highlights](#Highlights)
3. [Main Contributions](#Main-Contributions)
4. [Installation](#Installation)
5. [Run MedContext](#Run-MedContext)
6. [Results](#Results)
7. [Citation](#Citation)
8. [Contact](#Contact)
9. [Acknowledgements](#Acknowledgements)

<hr>

## Updates

* [June 18, 2024] Our paper is accepted at MICCAI 2024 (acceptance rate < 31%).
* [Feb 22, 2024] Code for UNETR is released.

## Highlights


> **Abstract:** *Volumetric medical segmentation is a critical component
of 3D medical image analysis that delineates different semantic regions. Deep neural networks have significantly
improved volumetric medical segmentation, but they generally require large-scale annotated data to achieve better
performance, which can be expensive and prohibitive to obtain. To address this limitation, existing works typically
perform transfer learning or design dedicated pretrainingfinetuning stages to learn representative features. However,
the mismatch between the source and target domain can
make it challenging to learn optimal representation for volumetric data, while the multi-stage training demands higher
compute as well as careful selection of stage-specific design choices. In contrast, we propose a universal training
framework called MedContext that is architecture-agnostic
and can be incorporated into any existing training framework for 3D medical segmentation. Our approach effectively learns self-supervised contextual cues jointly with
the supervised voxel segmentation task without requiring
large-scale annotated volumetric medical data or dedicated
pretraining-finetuning stages. The proposed approach induces contextual knowledge in the network by learning to
reconstruct the missing organ or parts of an organ in the
output segmentation space. The effectiveness of MedContext is validated across multiple 3D medical datasets and
four state-of-the-art model architectures. Our approach
demonstrates consistent gains in segmentation performance
across datasets and different architectures even in few-shot
data scenarios*
>
<hr>

## Main Contributions
* We propose a universal training framework called **MedContext** that is architecture-agnostic and can be incorporated into any existing training frame- work for 3D medical segmentation. 
* Our approach effectively learns self-supervised contextual cues jointly with the supervised voxel segmentation task without requiring large-scale annotated volumetric medical data or dedicated pretraining-finetuning stages. The proposed approach induces contextual knowledge in the network by learning to reconstruct the missing organ or parts of an organ in the output segmentation space.
* We validate the effectiveness of our approach across multiple 3D medical datasets and state-of-the-art model architectures. Our approach complements existing methods and improves segmentation performance in conventional as well as few-shot data scenarios


## Methodology
![](https://github.com/hananshafi/MedContext/blob/main/assets/3dmsr_main_diagram.png)



## Installation

```bash
# Create conda environment from yaml file
conda env create --name medcontext --file=environments.yml

# Activate the environment
conda activate medcontext
```

## Datasets
### BTCV

The BTCV data is from the [BTCV challenge dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752).

The dataset contains 13 abdominal organs including 1. Spleen 2. Right Kidney 3. Left Kideny 4. Gallbladder 5. Esophagus 6. Liver 7. Stomach 8. Aorta 9. IVC 10. Portal and Splenic Veins 11. Pancreas 12. Right adrenal gland 13. Left adrenal gland.

In this paper, we utilize 8 organs. Refer to our paper for further details.

Task: Segmentation

Modality: CT

Size: 30 3D volumes (18 Training + 12 Testing)

We provide the json file  that is used to train our models under ./datasets folder [here](https://github.com/hananshafi/MedContext/blob/main/UNETR/BTCV/dataset/dataset_18_12.json).

Please refer to [Setting up the datasets](https://github.com/282857341/nnFormer) on nnFormer repository for more details. Alternatively, you can download the preprocessed dataset for Synapse [here](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/abdelrahman_youssief_mbzuai_ac_ae/EbHDhSjkQW5Ak9SMPnGCyb8BOID98wdg3uUvQ0eNvTZ8RA?e=YVhfdg).

The dataset folders for Synapse should be organized as follows: 

```
./UNETR/BTCV/dataset/
  ├── unetr_pp_raw/
      ├── unetr_pp_raw_data/
           ├── Task02_Synapse/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
           ├── Task002_Synapse
       ├── unetr_pp_cropped_data/
           ├── Task002_Synapse
 ```

## Run MedContext
### Train UNETR on BTCV: 

```bash
cd UNETR/BTCV
python main.py --json_list dataset_18_12.json --val_every 100 --batch_size=1 --feature_size=32 --rank 0 --logdir=PATH/TO/OUTPUT/FOLDER --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 --save_checkpoint --data_dir=./dataset
```
Training support for other models and datasets will be released soon

### Test UNETR on BTCV: 
```bash
python test_8.py --infer_overlap=0.5 --json_list dataset_18_12.json --feature_size 32 --data_dir=./dataset --pretrained_model_name student_4000.pt --pretrained_dir='PATH/TO/SAVED/CHECKPOINT' --saved_checkpoint=ckpt
```
Change the --pretrained_model_name according to your saved checkpoint

### Train nnFormer on BTCV

```bash
cd nnFormer
DATASET_PATH=./UNETR/BTCV/dataset/

export PYTHONPATH=./
export RESULTS_FOLDER=PATH/TO/SAVE/CHECKPOINTS/
export nnFormer_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse
export nnFormer_raw_data_base="$DATASET_PATH"/unetr_pp_raw
python nnformer/run/run_training.py 3d_fullres nnFormerTrainerV2_nnformer_synapse 2 0 -c
```

## Contact
Should you have any questions, please contact at hanan.ghani@mbzuai.ac.ae

## Citation
If you use our work, please consider citing:
```bibtex 
@article{gani2024medcontext,
  title={MedContext: Learning Contextual Cues for Efficient Volumetric Medical Segmentation},
  author={Gani, Hanan and Naseer, Muzammal and Khan, Fahad and Khan, Salman},
  journal={arXiv preprint arXiv:2402.17725},
  year={2024}
}
```
## Acknowledgements
Our code is built on the repositories of  [MONAI](https://github.com/Project-MONAI/research-contributions). We thank them for their open-source implementation and instructions.
