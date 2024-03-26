# faceRecognitionApp
## Task: 
Build a Streamlit App where given a query face image, app should search, recommend and display similar faces in the database.
<br>
**Input:** Upload any face image among the 105 actors in the internet or test_dataset(which is completely different from training and validation dataset) or video where these 105 actors are present.
<br>
**Output:** If input is an image, identify the person among 105 classes(or actors) and also display all the similar faces available in the dataset with paths; otherwise tell that the person does not exist in the database. If the input is video, among multiple frames, select the one best frame of a person and identify the person among 105 classes(or actors) and also display all the similar faces available in the dataset with paths; otherwise tell that the person does not exist in the database.
<br>
## Process: 
#### 1. Dataset
The dataset is taken from Kaggle under Public Domain: [![Download KaggleDataset](https://img.shields.io/badge/Download-Dataset-blue)](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition?resource=download). This dataset contains 105 classes(actors) where each class consists of certain(not fixed) number of face images.
#### 2. Background
- Python 3.8.10
- Pytorch
- Refer requirements.txt file for libraries used with version
#### 3. Dataset Preparation
Separate out the whole dataset to Train and Test in the ratio of 80%:20% respectively. In addition to separating the whole dataset, for the purpose of standardization for training and process of query image during testing, each image will undergo face detection where the needed amount of face region is extracted by the bounding box and resize to 112\*112 dimension. This is implemented in prepare_split_dataset.py file and the output folder is named as "prepared_dataset".
#### 4. Train and Test a 105 Classes(Multi-class) Image Classifier
Using Pytorch, we will train ResNet50 to get the weights of the trained model in .pt file(resnet50_classifier.pt) so that we can use this .pt file's weights to get the feature embeddings of any image for Image Similarity task in the further process.<br>
**Implementation of training_resnet50_classifier.py:** As mentioned earlier, we will train a ResNet50 using "train" dataset in "prepared_dataset" folder for 50 epochs and output the Training details at each epoch ie Epoch number, train loss, test loss, train accuracy, test accuracy and epoch time in a csv file and store the csv file with the name classifier_training_details.csv in "outputs" folder; and also storing the weights file at each epoch in "weights" subfolder under "outputs" folder; with saving the graphs of train and val loss vs epochs and train and val accuracy vs epochs in a single image with the name classifier_training_plots.jpg in "outputs" folder. Xavier Initialization for initializing the weights, Categorical Cross Entropy as the Loss Function, Adam as the optimizer, Early stopping and Patience are also implemented.<br>
![Image - classifier_training_plots.jpg](outputs/classifier_training_plots.jpg)
**Implementation of testing_resnet50_classifier.py:** Using the final resnet50_classifier.pt file, I will predict the outputs for the "test" dataset in "prepared_dataset" folder and evaluate the model using Overall Accuracy, Average inference time accross all the images, Overall Precision, Recall and F1 Score. And plot the Confusion Matrix for 105 classes in jpg format and class-wise accuracy in bar graph in jpg format and save them in "outputs" folder with the names "test_classifier_confusion_matrix.jpg" and "test_classifier_classwise_accuracy.jpg" respectively.<br>
Performance details of the resnet50_classifier.pt on the Test Dataset:<br>
![Image - test_performance.png](outputs/test_performance.png)
Note: The results are not satisfying because I have less number of data, have not performed any data augmentation and have only run upto 20 epochs because each image's feature embedding is of size 2048 for which we need some GPU facility. So I am moving ahead with this model to showcase the end-to-end implementation. So going ahead, we'll not have good or have same accuracy in the further process.<br>
#### 5. Image Similarity Check using Cosine Similarity
**Phase I - Getting ready to get similarity_status of each pair of images in the test dataset:**<br> 
We will store the feature embeddings of each image(each of size 2048 using the trained resnet50_classifier.pt file) with respect to the class it belongs to serially in a csv file with the name "test_dataset_features.csv". Using "test_dataset_features.csv" file, I will create "actual_similarity_combinations.csv" file which will contain data regarding different combinations of image pairs in the test dataset in the following format: path-to-image1, feature-embeddings-of-image1, path-to-image2, feature-embeddings-of-image2, similarity_status. similarity_status=0 if image1 and image2 belong to different class and similarity_status=1 if image1 and image2 belong to same class. So there will br (number_of_test_dataset)C(2) ways of pairs of images with the similarity_status at each line in this csv file. The "actual_similarity_combinations.csv" file contains the actual similarity_status of every pair of images in the test dataset, using this we will calculate the Accuracy of the Face Recognotion from the predicted similarity_status as we go further. These things are implemented in "get_similarity_status.py" file.<br>
**Phase II - Choosing the right Cosine Similarity Score:**<br>
Firstly, I would like to rescale the cosine values from (-1, +1) to (0, 1) for more comfort. Now for different values of cosine similarity(from 0.1, incrementing 0.1 at each step), I want you to calculate FNMR score and store it in the csv file called "cs_fnmr.csv" so that I can select that cosine similarity value that has least FNMR score. These things are implemented in "get_cosine_fnmr.py" file.<br>
**Phase III:** 
#### 6. Streamlit App for Face Recognition





