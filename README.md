# VIB audio classification
Audio classification with variational information bottleneck (VIB)

## environment setting 
-python 3.7.4
-torch 1.7.1

## Feature set information

For this task, the dataset is built using 5252 samples from:


- [the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) dataset](https://zenodo.org/record/1188976#.YECqhC0Rr0p)

- [the Toronto emotional speech set (TESS) dataset](https://tspace.library.utoronto.ca/handle/1807/24487)

The classes the model wants to predict are the following: (0 = neutral, 1 = calm, 2 = happy, 3 = sad, 4 = angry, 5 = fearful, 6 = disgust, 7 = surprised). This dataset is skewed as there is not a calm class in TESS, hence there are less data for that particular class and this is evident when observing the classification report.
