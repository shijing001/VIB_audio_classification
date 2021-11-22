# VIB audio classification
Audio classification with variational information bottleneck (VIB)

## Dependencies
-python 3.7.4
-torch 1.7.1

## Feature set information

For this task, we use 3 datsets: emontiontoronto, urbansound8k, audioMNIST.

1. The emontiontoronto dataset is built using 5252 samples from:


- [the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) dataset](https://zenodo.org/record/1188976#.YECqhC0Rr0p)

- [the Toronto emotional speech set (TESS) dataset](https://tspace.library.utoronto.ca/handle/1807/24487)

The classes the model wants to predict are the following: (0 = neutral, 1 = calm, 2 = happy, 3 = sad, 4 = angry, 5 = fearful, 6 = disgust, 7 = surprised). This dataset is skewed as there is not a calm class in TESS, hence there are less data for that particular class and this is evident when observing the classification report.

2. urbansound8k : around 8000 samples of sounds from natural environment
3. audioMNIST : 3000 samples from 6 speaker prounce 0-9 in English, wiht 50 samples per speaker 

## Usage

With Pytorch>=1.7.0 environment, you can either run the Notebook (VIB_audio_classifier) or enter he following lines in terminal directly 

1. train over Emotion Toronto dataset:
python main.py --mode train --beta 1e-3 --data emotiontoronto --epoch 50 --lr 1.e-3 --K=64 --batch_size=32

(with default values all parameters unless specified)

2. test over Emotion Toronto dataset:
python main.py --mode test --beta 1e-3 --data emotiontoronto --epoch 50 --lr 1.e-3 --K=64 --batch_size=32 --load_ckpt best_acc.tar

## References
If you find this repository helpful, please cite our paper: 
Variational Information Bottleneck for Effective Low-resource Audio
Classification, Shijing Si, et al. 2021, https://www.isca-speech.org/archive/pdfs/interspeech_2021/si21_interspeech.pdf
