# Face Detector
I am building 2 types of Face Detectors:
1. By using Facenet by Google
2. By using Deepface by Facebook

Task: Given an input folder with loosely cropped face images, take each image and detect face in it in 112*112 dimension and keep it in output folder.

Functioning: For each face image, I wanted to detect the face, then calculate the centre ie (centre_x, centre_y) of the face detected region, then find the maximum between the width and height of the face detected region ie max_dim=max(w, h). I want to get the output image as a square image with dimension (max_dim*max_dim) by: (i)If the max_dim=w, extend the height equally from top and bottom or (ii) If the max_dim=h, extend the height equally from left and right; and do black padding if the measurements go beyond 0 or image.shape so that we maintain the same centre ie (centre_x, centre_y) of face detected region for our output image. Then resize the output image to (112\*112) for standardization.

Observations:
(i) Facenet and Deepface has given me equal accuracy in Face Detection task. Performance wise both are almost same.
(ii) Facenet is faster than Deepface because Facenet is a lighter model, hence face detected output's resolution is almost half of deepface's face detected output's resolution.

Inference: I will choosing Facenet as my Face Detector for further process of this project.
