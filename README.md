# Multiple-Instance-Learning-MIL-with-Keras-using-clear-cell-renal-cell-carcinoma-dataset
MIL project with Keras for classification histology images with clear cell renal cell carcinoma
Article: 
Medium - https://medium.com/@nik88888888888/multiple-instance-learning-mil-using-keras-with-clear-cell-renal-cell-carcinoma-histology-b812c9e0241d

## Intro
Taking into account the relevance of Multiple instance learning (MIL) and, in particular, the advantages of this method for the analysis of histological images, I decided to try to train models in order to classify bags of instances into those, that contain only normal tissues (negative class) and those, in which images are found with clear cell renal cell carcinoma (positive class).
## Dataset
Datasets containing 500, 1000 and 2000 bags of instances were used to train the models. The ratio of positive (containing images of clear cell renal cell carcinoma) to negative (containing only normal tissues) was 1:1. Each set contained 40 colorful .jpeg images with a resolution of 256x256 pixels, obtained from full-slide images from the CPTAC-CCRCC study (WSI is freely available on the Cancer Imaging Archive website). In the positive sets, 20 out of 40 images were with clear cell renal cell carcinoma.

![классы](https://user-images.githubusercontent.com/55003096/198854348-d0e164e8-abef-4a0d-bd29-6e65db78f697.png)

## Model
Model structure - https://github.com/bluewhale28/Multiple-Instance-Learning-MIL-with-Keras-using-clear-cell-renal-cell-carcinoma-dataset/blob/main/SimpleModel.py

## Training results
![results](https://user-images.githubusercontent.com/55003096/198854324-6f820e7a-1fae-4a6a-b588-60802edd772d.png)
## Test results
![результаты](https://user-images.githubusercontent.com/55003096/198854336-e46e6291-4665-4855-be61-c099b0662b55.png)
### Confusion matrix
![conf_40_20](https://user-images.githubusercontent.com/55003096/198854373-98c7c2d2-813b-4158-9add-8ae07945228e.png)
![40_10](https://user-images.githubusercontent.com/55003096/198854376-a302583d-19eb-4908-b923-e38533daec22.png)
![40_5](https://user-images.githubusercontent.com/55003096/198854389-dcf7686d-05fb-45ea-bd9a-3c9a41608690.png)
![40_1](https://user-images.githubusercontent.com/55003096/198854392-8111b5f3-a7b5-4190-ba02-3c328a2f5ead.png)

