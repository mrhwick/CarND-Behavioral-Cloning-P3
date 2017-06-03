# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center-lane.jpg "Center Lane Driving"
[image2]: ./examples/recovery-right.jpg "Recovery drive from the right"
[image3]: ./examples/recovery-left.jpg "Recovery drive from the left"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
When running the Udacity provided simulator in autonomous mode and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for loading the necessary data points, as well as training and saving the neural netowrk model for the steering agent. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 138-143). 

The model includes RELU activation layers within the convolution network to introduce nonlinearity (code line 138-143), and the data is normalized in the model using a Keras lambda layer (code line 136). In addition, the model contains a cropping layer to reduce the amount of noisy and unnecessary data to be processed, which is not representative of the road.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 142, 146, and 148). 

The model was trained and validated on a data set collected to ensure that the model was not overfitting (code line 14-114). This data was collected using the simulator provided by Udacity, using both tracks provided to ensure generalization. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model uses an adam optimizer, so the learning rate was not tuned manually (model.py line 157).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I collected training data for driving down the center of the road with smooth turns around the curves of each track, clockwise and counterclockwise. These laps were collected at both maximum speed (30 MPH) and a minimum speed (10 MPH). Then I collected some "recovery" data, illustrating to the model the necessity of moving back to the center of the track if the car strays too far to the side.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use the convolution neural network model published by Nvidia, which has been used to train the steering agent from a forward-facing camera. This model fits the problem we are trying to solve exactly, so it was a good choice for this project.

To combat the possibilty of overfitting, I modified the model by adding Keras Dropout layers betwee the last two convolutional layers and between the first three layers of the fully connected network.

I also added the normalization and cropping layers myself, which I do not believe match the published Nvidia model. I simply used the color images as is, because I believe that the simulator and drive.py provide this to the network. Nvidia apparently converted their images into a different color channel as well, which may have improved the performance of my model.

Once I had recreated the Nvidia network in Keras with some adjustments of my own, I was ready to attempt a training session, using AWS GPU instances to speed up the process. I initially did not see my network converging during training. This was a result of a bug in my training set generator that shuffled the training inputs and the labels associated with those inputs separately from one another. This lead to my network converging near 0.0325 loss during each training step, which is 1/32 (essentially the network was simply guessing at each training step between 32 possible options and was correct 1/32 of the time, leading to this accuracy).

Once I corrected this bug, I attempted another training session. This time, my model reliably increased in accuracy on both the training and validation sets provided. After two training sessions that took about 45 minutes on the AWS GPU instance, the model was able to reliably drive the car around both tracks. At higher driving speeds, the steering agent tends to be a bit more shaky, which I believe is related to the "recovery" training data which I provided to the model.

#### 2. Final Model Architecture

The final model architecture (model.py lines 136-150) consisted of a convolution neural network with the following layers and layer sizes:

```python
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.4))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Dense(1))
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded a few laps on track one using center lane driving. Here is an example image of center lane driving on track one:

![An example of center lane driving on a straightaway][image1]

I performed similar center-lane driving on track two as well as driving both tracks in reverse to ensure generalization of the model to different kinds of tracks.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would recognize the boundaries of the road and understand that it was expected to turn back towards the center from those boundaries. These images show what a recovery looks like on both the right and left sides.

![A recovery drive from the right side of the track][image2]
![A recovery drive from the left side of the track][image3]

To augment the data sat, I also flipped images and angles so that the total number of "tracks" the steering agent was trained on would be increased to 8 (clockwise and counter clockwise, unreversed and reversed, for both of the provided tracks).

After the collection process, I had about 22,000 data points to work with, which I then split into training and validation sets of ~16,500 and ~5,500 respectively. After augmentation, the total number of training set examples totalled ~99,500 examples. This set included adjusted examples from the center camera, both the left and right camera, as well as the flipped versions of each of these examples.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting, which was very helpful in evaluating whether my model was incorrect due to a bug or due to my provided training data. I trained the model for 15 epochs during each training session, which allowed me to see good fitting to the provided data without overfitting.

I performed one additional step to determine whether errors were the result of bugs or the result of bad training data. I crafted an extremely small data set of hand-picked examples exhibiting the three extreme steering positions: hard left, hard right, and center. I then trained my model on this set of three images and allowed the model to overfit to the data. If the model was capable of overfitting to this tiny dataset, I assumed that the problem was with my data rather than my algorithm or model. Using this method exposed a bug in my generator code that was shuffling the data incorrectly before training.