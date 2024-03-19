# faceRecognitionApp
## Task: 
Build a Streamlit App where given a query face image, app should search, recommend and display similar faces in the database.
Input: Upload any face image among the 105 actors in the internet or test_dataset(which is completely different from training and validation dataset) or video where these 105 actors are present.
Output: If input is an image, identify the person among 105 classes(or actors) and also display all the similar faces available in the dataset with paths; otherwise tell that the person does not exist in the database. If the input is video, among multiple frames, select the one best frame of a person and identify the person among 105 classes(or actors) and also display all the similar faces available in the dataset with paths; otherwise tell that the person does not exist in the database.
#### Training: 
1. Data Preparation:
The dataset is taken from Kaggle under Public Domain: [![Download KaggleDataset](https://img.shields.io/badge/Download-Dataset-blue)](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition?resource=download). 
This dataset contains 105 classes(actors) where each class consists of certain(not fixed) number of face images. 
2. 



