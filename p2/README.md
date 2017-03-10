#**Traffic Sign Recognition** 

Final Test Set Accuracy : 96.5 %


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

[image9]: ./examples/distribution1.jpg "Visualization"
[image10]: ./examples/distribution2.jpg "Grayscaling"
[image11]: ./examples/sign_before.jpg "Random Noise"
[image12]: ./examples/sign_after.jpg "Traffic Sign 1"
[image13]: ./examples/split_shuffle.jpg "Traffic Sign 2"
[image14]: ./examples/fake_image.jpg "Traffic Sign 3"
[image15]: ./examples/final_images.jpg "Traffic Sign 4"
[image16]: ./examples/placeholder.png "Traffic Sign 5"

[image17]: ./examples/web_signs.jpg "Traffic Sign 4"
[image18]: ./examples/model_selection.jpg "Traffic Sign 4"
[image19]: ./data/2.jpg "Traffic Sign 4"
[image20]: ./data/38.jpg "Traffic Sign 4"
[image21]: ./data/40.jpg "Traffic Sign 4"



You're reading it! and here is a link to my [project code](https://github.com/jongchul/self-driving-car-nd/blob/master/p2/Traffic_Sign_Classifier_submit.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

 I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 39209
* The size of test set is ? 12630
* The shape of a traffic sign image is ? (32, 32, 3)
* The number of unique classes/labels in the data set is ? 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 7~9th code cells of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart.
The x-axis represents each traffic sign. The y-axis represents a number of samples for each traffic sign. 

![alt text][image9]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 2nd ~ 5th code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because RGB color space has wide variation among same object images than gray scale image space. I choose to use only Y value 
in YCrCb color space. (https://en.wikipedia.org/wiki/YCbCr) 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image11]

![alt text][image12]


As a last step, I normalized the image data, using skimage package. 
(http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist) 

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the 24th code cell of the IPython notebook.  


![alt text][image13]

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using sklearn python package (http://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html)

My final training set had 42,356 number of images. My validation set and test set had 18,153 and 12,630 number of images.

The 12th code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because some traffic sign has very small data set and it can cause unbalanced batches to the model during training.  To add more data to the the data set, I selected images which has less than 400 number of images  and made 5 copies of images. And I used random rotation within 10 degrees and translation within 5 pixels in each X and Y axis because similar images are better for testing.  

![alt text][image14]

Here is an example of an augmented images:

![alt text][image10]

![alt text][image15]

 

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 27th cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1  image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| DROPOUT				| 0.6											|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x64	|
| RELU					|												|
| DROPOUT				| 0.6											|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64 					|
| Fully connected		| 12288x1200       								|
| RELU					|												|
| DROPOUT				| 0.7											|
| Fully connected		| 1200x43         								|
| Softmax				|        										|



####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 29th and 34th cell of the ipython notebook. 

To train the model, I used an 
- Adam optimizer
- batch size : 128
- number of epochs : 100
- learning rate : 0.001

I used 128 batch size because 144, (total number of images, 60,509/ the most small number of image, 420) is similar to 128. I followed common parameter values for Adam optimizer and learning rate. Since my dataset is small, I used 100 epochs for training.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I used 4 layer CNN architecture. Below images are error images during validation process. I collected these error images randomly using evaluation data set to invesitigte my model. 

The code for debugging the model is located in different file.
Traffic_Sign_Classifier_submit_revised.ipynb.(https://github.com/jongchul/self-driving-car-nd/blob/master/p2/Traffic_Sign_Classifier_submit_revised.ipynb)

The code for debugging the model is located in the 299th and 300th cell of the ipython notebook,

![alt text][image18]

The code for calculating the accuracy of the model is located in the 34th  cell of the Ipython notebook.

My final model results were:
* training set at last batch accuracy of ? 100%
* validation set at last batch accuracy of ? 100% 
* test set accuracy of ? 96.5 %



If an iterative approach was chosen:

- I tested 4 layer CNN without L2 regularization very first and found out that regularization can help 1~2% accuracy boost. I added dropout layers betweein convolution and fully connected layer for 
further regularization. 

- I made 2 additional dataset. 

1) X_train_original_fake : 5 copies of images whose number are below 400.

2) X_train_original_fake_second : X_train_original_fake + 3 copies of images whose number are below 1000.

The first additional dataset works better than the second addtional dataset. I think that the first one, X_train_original_fake, can provide more balanced training batch data to the model than the second one.

I calcuated total loss of training set at each epoch and validation set at every 10 epoch. And I finally calculated accuracy of test set at the final epoch. These total loss of training and validation data shows that my model is getting better during training.
  
The code for calculating the total loss of the training and validation is located in the 33th~34th  cell of the Ipython notebook.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image17] 

The stop sign image might be difficult to classify because the location of third one is skewed and noise.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 40th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection	| Right-of-way at the next intersection		| 
| Speed limit (50km/h)  	 			| Speed limit (50km/h)   					|
| Stop									| Priority road								|
| Roundabout mandatory					| Roundabout mandatory						|
| Yield									| Yield										| 


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 96.5%, since clear images without noise show correct predictions.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 40th cell of the Ipython notebook.

For the most of images, the model is relatively sure.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| Right-of-way at the next intersection    									| 
| .0     				| Speed limit (20km/h) 										|
| .0					| Speed limit (30km/h)											|
| .0	      			| Speed limit (50km/h)					 				|
| .0				    | Speed limit (60km/h)     							|


For the second image 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| Speed limit (50km/h)     									| 
| .0     				| Speed limit (120km/h) 										|
| .0					| Speed limit (80km/h)											|
| .0	      			| Speed limit (20km/h)					 				|
| .0				    | Speed limit (30km/h)     							|

For the third image 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9601        		| Priority road     									| 
| 0.0399   				| Stop 										|
| .0					| Road work											|
| .0	      			| Yield					 				|
| .0				    | Speed limit (100km/h)     							|

For the fourth image 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Roundabout mandatory    									| 
| .0   					| Speed limit (20km/h) 										|
| .0					| Speed limit (30km/h)										|
| .0	      			| Speed limit (50km/h)					 				|
| .0				    | Speed limit (60km/h)     							|


For the fifth image 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Yield   									| 
| .0   					| Road work 										|
| .0					| Priority road										|
| .0	      			| Roundabout mandatory					 				|
| .0				    | Double curve     							|