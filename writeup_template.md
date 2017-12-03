#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/image1.jpg "recorded data"
[image2]: ./examples/image2.jpg "left recorded data"
[image3]: ./examples/image3.jpg "right recorded data"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* Nvidia_e30_b8.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model uses Nvidia CNN architecture for self-driving cars. (Model in lines 101-116)

The model activates RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road: the sample data provided by udacity and additional recorded laps.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to test different models like LeNet, Nvidia..

My first step was to use the simple initial model (lines 69-75), but it failed to predict correct angles.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

Then I tried the LeNet model to have better prediction for angles. 
Then tried the Nvidia model and modified it by removing the last convolution layer.
The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 101-116) consisted of a convolution neural network with the following layers and layer sizes:

| Layer (Type ) 		|     layer size	        					| 
|:---------------------:|:---------------------------------------------:| 
| Cropping2D         	| 64, 64, 3         							| 
| Lambda            	| 64, 64, 3          							| 
| Convolution          	| 24, (5, 5)        							|
| Convolution          	| 36, (5, 5)        							|
| Convolution          	| 48, (5, 5)        							|
| Convolution          	| 64, (3, 3)        							|
| Convolution          	| 64, (3, 3)        							| 
| Flatten            	|                   							| 
| Dropout(0.5)         	|                   							| 
| Dense             	| 100                  							| 
| Dense             	| 50                  							| 
| Dense             	| 10                  							| 
| Dense             	| 1                  							| 


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road

![alt text][image2]
![alt text][image3]

Then I repeated this process on track two in order to get more data points.

I finally randomly shuffled the data set and put 15% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 30.
