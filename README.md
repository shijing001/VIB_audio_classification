# VIB audio classification
Audio classification with variational information bottleneck (VIB)

## Dependencies
-python 3.7.4
-torch 1.7.1

## Feature set information

For this task, the dataset is built using 5252 samples from:


- [the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) dataset](https://zenodo.org/record/1188976#.YECqhC0Rr0p)

- [the Toronto emotional speech set (TESS) dataset](https://tspace.library.utoronto.ca/handle/1807/24487)

The classes the model wants to predict are the following: (0 = neutral, 1 = calm, 2 = happy, 3 = sad, 4 = angry, 5 = fearful, 6 = disgust, 7 = surprised). This dataset is skewed as there is not a calm class in TESS, hence there are less data for that particular class and this is evident when observing the classification report.

## Usage

For the VIB audio classifier, You can either run the Notebook or enter he following lines in terminal directly 

1. train (with default values all parameters unless specified)
python main.py --mode train --beta 1e-3 --env_name [NAME]

2. test
python main.py --mode test --env_name [NAME] --load_ckpt best_acc.tar

## References

1. Deep Variational Information Bottleneck, Alemi et al. [paper] (https://arxiv.org/abs/1612.00410)
2. Variational Information Bottleneck for Effective Low-Resource Fine-Tuning, Karimi et al. [paper](https://openreview.net/forum?id=kvhzKz-_DMF)
