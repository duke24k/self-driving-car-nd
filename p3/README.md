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

**Model Architecture**

Instead of using the NVIDIA architecture from their paper (5 convolutional layers, use convolutional strides rather than pooling, 3 fully connected layers) or the comma.ai architecture (3 convolutional layers, use convolutional strides rather than pooling, 1 fully connected layer), I ended up 4 convolutional layers and 2 fully connected layers and I decided to keep max pooling as a way of allowing the model to learn generailized behavior.


My model consists of a convolution neural network with 5x5, 3x3 filter sizes and depths of 24, 36, 48, 64. 
(The code for this step is contained in the 2th code cells of the IPython notebook, p3_final.ipynb) 
The model includes ELU layers to introduce nonlinearity and the data is cropped and normalized in the model using a Keras lambda layer. 

In order to reduce overfitting, the model contains maxpooling in cnn layers and dropout in fcn layers  

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually.

![alt text][image1]

---
**Data Downsampling**

I used udacity data to train my model. 0.35, steering anglie shift value, was used for left and right images. 
(The code for this step is contained from the third code cell of the IPython notebook, p3_final.ipynb) 


The data is highly toward 0, 0.35, -0.35, steering angles. I did downsampling work as below. 
downsampling results were slightly different and I tried to make mean value of downsampled data similar to mean value of original data.


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

---
**Preprocessing**

I used udacity data to train my model. Udacity data consists of center, left, right side camera images. 
I used 0.35, -0.35 steering angle offset for left and right side camera images.


![alt text][image6]
![alt text][image7]
![alt text][image8]


The model did not run after the first training. I trained model several times in the way of Training section. Finally my model ran on the first track except only one problem area. I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover on the problem area. These images show what a recovery looks like pulling away from the wall. 

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I did
- convert color space to YUV
- random darken
- vertical shift only (with udacity data)
- vertiacal and horizontal shift (with recovery data. manupulating more data with small data.)
- horizontal flipping
- crop


![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]


(The code for this step is contained from the 14th to 16th code cell of the IPython notebook, p3_final.ipynb)

---
**Training**

I used udacity data and recovery data to train my model. First I split udacity data into train and validation set. I used 256 batch size and 2~3 epochs. The result were slightly  different. Some models ran better than other models. I saved the model which showed better performance on the simulator with versioning. And I restored the model and trained it again with recovery data in similar way to train the model with udacity data. If the retrained model looked better, I mean it keep distance from the wall on the problem area, I saved it. I tried several times and finally my model run on the 1st track without problem. 

To check downsampling result, I used seperated ipython notebook file. 

- p3_final.ipynb 
- p3_patch.ipynb

(The code for this step is contained from the 18th to 20th code cell of the IPython notebook, p3_final.ipynb)

---
**Code submission**

My project includes the following files:

| File         				|     Description	        					  					  								| 
|:-------------------------:|:-------------------------------------------------------------------------------------------------:| 
| p3_final.ipynb   			| create and train the model using udacity data   													| 
| p3_patch.ipynb    	 	| create and train the model using recovery data which I additionally collected 					|
| drive_yuv.py				| driving the car in autonomous mode																|
  model_v3_patch_v6.h5		| model																								|
| run1.mp4					| driving record																					|
| README.md	 	 	  	| 																									|


* p3_final.ipynb and p3_patch.ipynb are same. I used different directories and different sub-sampling parameters for different data.

* I added below codes to drive.py file to resize image and convert color space to YUV.
```sh                                                     
     resized = cv2.resize( np.asarray(image), (320, 160))
     file = cv2.cvtColor( resized , cv2.COLOR_BGR2YUV)
```

| Driving Record       		|     Download Link 	        	  					  											| 
|:-------------------------:|:-------------------------------------------------------------------------------------------------:| 
| run1.mp4      			| https://github.com/jongchul/self-driving-car-nd/blob/master/p3/run1.mp4							| 
|     	 					| https://github.com/jongchul/self-driving-car-nd/blob/master/p3/movie.mp4 							|


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










