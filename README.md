# MedContext:  Learning Contextual Cues for Efficient Volumetric Medical Segmentation
(full code and all the models will be released soon!)

[Hanan Gani<sup>1</sup>](https://hananshafi.github.io/), [Muzammal Naseer<sup>1</sup>](https://muzammal-naseer.com/), [Fahad Khan<sup>1,2</sup>](https://sites.google.com/view/fahadkhans/home), [Salman Khan<sup>1,3</sup>](https://salman-h-khan.github.io/)

<sup>1</sup>Mohamed Bin Zayed University of AI      <sup>2</sup>Linkoping University      <sup>3</sup>Australian National University

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()

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

* Code for UNETR is released. [Feb 22, 2024]

## Highlights


> **Abstract:** * *
>
<hr>

## Main Contributions
* We propose a universal training framework called **MedContext** that is architecture-agnostic and can be incorporated into any existing training frame- work for 3D medical segmentation. 
* Our approach effectively learns self-supervised contextual cues jointly with the supervised voxel segmentation task without requiring large-scale annotated volumetric medical data or dedicated pretraining-finetuning stages. The proposed approach induces contextual knowledge in the network by learning to reconstruct the missing organ or parts of an organ in the output segmentation space.
* We validate the effectiveness of our approach across multiple 3D medical datasets and state-of-the-art model architectures. Our approach complements existing methods and improves segmentation performance in conventional as well as few-shot data scenarios


## Methodology
![main-figure](https://github.com/hananshafi/MedContext/blob/main/assets/3dmsr_main_diagram.png)



## Installation

```bash
# Create a conda environment
conda create -n medcontext python==3.8

# Activate the environment
conda activate medcontext

# Install requirements
pip install -r requirements.txt
```

## Run MedContext



## Contact
Should you have any questions, please contact at hanan.ghani@mbzuai.ac.ae

## Citation
If you use our work, please consider citing:
```bibtex 

```
## Acknowledgements
Our code is built on the repositories of  [MONAI](https://github.com/Project-MONAI/research-contributions). We thank them for their open-source implementation and instructions.
