Implementation paper [A Simple Baseline for Multi-Object Tracking](https://arxiv.org/pdf/2004.01888v4.pdf)

[![Python](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7%20%7C%203.8-blue)](https://www.python.org)
[![Pytorch](https://img.shields.io/badge/pytorch-1.6.0-red)](https://pytorch.org/)
[![conda](https://img.shields.io/badge/conda-green)](https://docs.conda.io/en/latest/miniconda.html)

## Tracking Performances
### Results on MOT challenge test set
| Dataset | MOTA | IDF1 | IDS  | MT    | ML    | FPS  |
|---------|------|------|------|-------|-------|------|
| 2DMOT15 | 59.0 | 62.2 | 582  | 45.6% | 11.5% | 30.5 |
| MOT16   | 68.7 | 70.4 | 953  | 39.5% | 19.0% | 25.9 |
| MOT17   | 67.5 | 69.8 | 2868 | 37.7% | 20.8% | 25.9 |
| MOT20   | 58.7 | 63.7 | 6013 | 66.3% | 8.5%  | 13.2 |

## Installation
 * Clone this repo
 * Install dependencies. I use miniconda3 for python 3.8 and pytorch 1.6.0
 * [DCNv2](https://github.com/CharlesShang/DCNv2) is used for bakcbone network
 ```
 # create environment
 conda create -n <env_name>
 conda activate <env_name>
 conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

 # return to the root directory
 pip install -r requirements.txt
 cd src/lib/models/networks/DCNv2_new sh make.sh
 ```

## Data Preparation


## Demo

### References
 * [MOT](https://paperswithcode.com/paper/a-simple-baseline-for-multi-object-tracking)
 * Significant amounts of code are borrowed from the [offical implemtation code](https://github.com/microsoft/FairMOT)