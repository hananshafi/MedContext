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
* **Scene Blueprints:** we present a novel approach leveraging Large Language Models (LLMs)
to extract critical components from text prompts, including bounding box coordinates for foreground objects, detailed textual descriptions for individual objects,
and a succinct background context. Utilizing bounding 
* **Global Scene Generation:** Utilzing the bounding box layout and genralized background prompt, we generate an initial image using Layout-to-Image generator.
* **Iterative Refinement Scheme :** Given the initial image, our proposed refinement mechanism iteratively evaluates and refines the box-level content of each object to align
them with their textual descriptions, recomposing objects as needed to ensure consistency.


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
