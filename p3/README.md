**Behavioral Cloning Project**

[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/recovery1.jpg "Recovery Image"
[image4]: ./examples/recovery2.jpg "Recovery Image"
[image5]: ./examples/recovery3.jpg "Recovery Image"

[image6]: ./examples/udacity1.jpg "Recovery Image"
[image7]: ./examples/udacity2.jpg "Recovery Image"
[image8]: ./examples/udacity3.jpg "Recovery Image"

[image9]: ./examples/raw.jpg "Recovery Image"
[image10]: ./examples/yuv.jpg "Recovery Image"
[image11]: ./examples/flip.jpg "Recovery Image"
[image12]: ./examples/final.jpg "Recovery Image"


[image13]: ./examples/subsample1.png "Recovery Image"
[image14]: ./examples/subsample2.png "Recovery Image"





---

###Model Architecture

Instead of using the NVIDIA architecture from their paper (5 convolutional layers, use convolutional strides rather than pooling, 3 fully connected layers) or the comma.ai architecture (3 convolutional layers, use convolutional strides rather than pooling, 1 fully connected layer), I ended up 4 convolutional layers and 2 fully connected layers and I decided to keep max pooling as a way of allowing the model to learn generailized behavior.


My model consists of a convolution neural network with 5x5, 3x3 filter sizes and depths of 24, 36, 48, 64. 
(The code for this step is contained in the 2th code cells of the IPython notebook, p3_final.ipynb) 
The model includes ELU layers to introduce nonlinearity and the data is cropped and normalized in the model using a Keras lambda layer. 

In order to reduce overfitting, the model contains maxpooling in cnn layers and dropout in fcn layers  

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually.

![alt text][image1]


###Data Downsampling

I used udacity data to train my model. 35 degree shift is used for left and right images. 
The data is highly toward 0, 35, -35, steering angles. I did downsampling work as below. 
downsampling result is slightly different and I tried to make similar mean value of original data.


1. udacity data
- total data points 24108 
- mean value of the data 0.00406964406483 

2. downsampling
- 23 bins 
- target value : 0.5
- 1/ (each bin value / target)

3. additional downsampling : centered bin reszie with scale factor 
-  left_cut, right_cut, center_cut(scale factor)


![alt text][image13]
![alt text][image14]

(The code for this step is contained from the 7th to 13th  code cell of the IPython notebook, p3_final.ipynb) 

reference:
https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project


####Preprocessing

I used udacity data to train my model. Udacity data consists of center, left, right side camera images. 
I used 35, -35 offset for left and right side camera images.


![alt text][image6]
![alt text][image7]
![alt text][image8]


The model did not run after the first training. I trained my model several times and made the model run on the first track except only one problem area. I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover. These images show what a recovery looks like pulling away from the wall. 

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I did
- convert color space to YUV
- random darken
- vertical shift and/or horizontal shift
- horizontal flipping
- crop


![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]




####Training 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.


###Code submission
My project includes the following files:

| File         				|     Description	        					  					  								| 
|:-------------------------:|:-------------------------------------------------------------------------------------------------:| 
| p3_final.ipynb   			| create and train the model using udacity data   													| 
| p3_patch.ipynb    	 	| create and train the model using recovery data which I additionally collected 					|
| drive_yuv.py				| driving the car in autonomous mode																|
| model_v3_patch_v6.h5 		| model 																							|
|  README.md	 	 	  	| 																									|


* p3_final.ipynb and p3_patch.ipynb are same. I use different directories and different sub-sampling parameters for different data.

* drive_yuv.py is same to drive.py which udacity provides. I added below codes to resize image and convert color space to YUV.
```sh                                                     
     resized = cv2.resize( np.asarray(image), (320, 160))
     file = cv2.cvtColor( resized , cv2.COLOR_BGR2YUV)
```

* Model
  
| Model        				|     Download Link	        					  					  										| 
|:-------------------------:|:-------------------------------------------------------------------------------------------------:| 
| model_v3_patch_v6.h5    	| https://drive.google.com/open?id=0B3qIpWd3o2CxeDdxaDlzTHBhOHM										| 
|     	 					| https://drive.google.com/file/d/0B3qIpWd3o2CxcmZWTVNvOW9KS28/view?usp=sharing 					|
  

I used the 1st track to test.
  - Screen resolution : 800*600  
  - graphic quailtiy : Simple

the car can be driven autonomously around the 1st track by executing 
```sh
python drive_yuv.py model_v3_patch_v6.h5
```

| Model        				|     Record Download Link	        	  					  										| 
|:-------------------------:|:-------------------------------------------------------------------------------------------------:| 
| model_v3_patch_v6.h5    	| https://github.com/jongchul/self-driving-car-nd/blob/master/p3/run1.mp4							| 
|     	 					| https://github.com/jongchul/self-driving-car-nd/blob/master/p3/movie.mp4 							|






