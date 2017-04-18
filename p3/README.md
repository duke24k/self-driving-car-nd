**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/recovery1.jpg "Recovery Image"
[image4]: ./examples/recovery2.jpg "Recovery Image"
[image5]: ./examples/recovery3.jpg "Recovery Image"

[image6]: ./examples/udacity1.jpg "Recovery Image"
[image7]: ./examples/udacity2.jpg "Recovery Image"
[image8]: ./examples/udacity3.jpg "Recovery Image"

[image9]: ./examples/raw.png "Recovery Image"
[image10]: ./examples/yuv.png "Recovery Image"
[image11]: ./examples/flip.png "Recovery Image"
[image12]: ./examples/final.png "Recovery Image"



---


My project includes the following files:

| File         				|     Description	        					  					  				| 
|:-------------------------:|:---------------------------------------------------------------------------------:| 
| p3_final.ipynb   			| create and train the model using udacity data   									| 
| p3_patch.ipynb    	 	| create and train the model using recovery data which I additionally collected 	|
| drive_yuv.py				| driving the car in autonomous mode												|
| model_v3_patch_v6.h5 		| model 																			|
|  README.md	 	 	  	| 																					|


* p3_final.ipynb and p3_patch.ipynb are same except some part sub-sampling logic, which I use different directories and different 
  sub-sampling parameters for different data.

* drive_yuv.py is same to drive.py which udacity provides. I added below codes to resize image and convert color space to YUV.
                                                     
     resized = cv2.resize( np.asarray(image), (320, 160))
     file = cv2.cvtColor( resized , cv2.COLOR_BGR2YUV)

* model_v3_patch_v6.h5 
  : https://drive.google.com/open?id=0B3qIpWd3o2CxeDdxaDlzTHBhOHM
  additional model
  : https://drive.google.com/file/d/0B3qIpWd3o2CxcmZWTVNvOW9KS28/view?usp=sharing
  

I used the 1st track to test.
  - Screen resolution : 800*600  
  - graphic quailtiy : Simple

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the 1st track by executing 
```sh
python drive_yuv.py model_v3_patch_v6.h5
```

https://github.com/jongchul/self-driving-car-nd/blob/master/p3/run1.mp4
#https://github.com/jongchul/self-driving-car-nd/blob/master/p3/movie.mp4
  


###Model Architecture and Training Strategy

My model consists of a convolution neural network with 5x5, 3x3 filter sizes and depths of 24, 36, 48, 64. 
The code for this step is contained in the 2th code cells of the IPython notebook, p3_final.ipynb,p3_patch.ipynb) 
The model includes ELU layers to introduce nonlinearity and the data is cropped and normalized in the model using a Keras lambda layer. 

In order to reduce overfitting, the model contains maxpooling in cnn layers and dropout in fcn layers  

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually.


####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The NVIDIA architecture from their paper (5 convolutional layers, use convolutional strides rather than pooling, 3 fully connected layers)
The comma.ai architecture (3 convolutional layers, use convolutional strides rather than pooling, 1 fully connected layer)
I ended up 4 convolutional layers and 2 fully connected layers and I decided to keep max pooling as a way of allowing the model to learn generailized behavior.

At the end of the process, the vehicle is able to drive autonomously around the 1st track without leaving the road.

https://github.com/jongchul/self-driving-car-nd/blob/master/p3/run1.mp4

#https://github.com/jongchul/self-driving-car-nd/blob/master/p3/movie.mp4

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image6]
![alt text][image7]
![alt text][image8]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
