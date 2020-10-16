# Mnist-Classification
 Here a Kaggle problem of digit recognizer is taken. The Aim is to use the maximum accuracy by using only limited no of traing data. nk
 
 contest link: [Click](https://www.kaggle.com/c/digit-recognizer)
 
# Overview

Here is the initial outline of sloving the problem.
### 1. Load the train and test data.<br>

### 2. Check for that the Data is balenced or not.<br>

### 3. Scale the data (MinMaxScaler/StandardScaler).<br>
   a. MinMaxScaler- It map the brightness of pixel between 0 to 1 range.<br>
   b. StandardScaler- It map the brightness of pixels between -1 to 1 range.<br><br>
   Here both scaling has been used but the result of MinMaxScaler is better as compared to StandardScaler.It also look reasonable because it map the brightness to 0 (darkness)         to 1 (brightness or white) 
   
### 4. Reshape data in images.<br>

### 5. Build the Model.<br>
In this problem the best model suited is CNN as it work well for the image data. Initially make a simple model and than iterate over it to make the better model. Since CNN consist of two parts-- 1) feature extraction(Convolution part) 2) Classificatin part.

### . Train the Model.<br>

### 7. Hyper-parmeter Tuning<br>

### 8. Predict the accuracy<br>.

This Give us a base CNN for our Job.


