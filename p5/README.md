**Vehicle Detection and Tracking.**


[//]: # (Image References)
[image0]: ./examples/car_hog.png
[image1]: ./examples/notcar_hog.png

[image2]: ./examples/svm_para1.png
[image3]: ./examples/svm_para2.png

[image4]: ./examples/window.png


[image5]: ./examples/example_image.png

[image6]: ./examples/FP.png

[image7]: ./examples/FP2.png

[video1]: ./base_v19.mp4


---

**I. Histogram of Oriented Gradients (HOG)**

**extracting HOG features from the training images.**

 The code for this step is contained from the second code cell to the 5th cell of the IPython notebook file, svc_train.ipynb. 
 
 I started by reading in all the vehicle and non-vehicle images. I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. Here is an example using the `YCrCb` color space with 1 channel and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image0]
![alt text][image1]


**final choice of HOG parameters.**

I tried various combinations of parameters to detect cars on the test images. Here is a example how I tested hog parameters.

![alt text][image2]


**training a classifier** 

1. Format features using np.vstack and StandardScaler().
2. Split data into shuffled training and test sets
3. Train linear SVM using sklearn.svm.LinearSVC().
 
 The code for this step is contained from the second code cell to the 6th cell of the IPython notebook file, svc_train.ipynb. 


**II.Sliding Window Search** 

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this 

1. pyramid = [
           ((64, 64),  [400, 500]),
           ((96, 96),  [400, 500]),
           ((128, 128),[450, 578])
      ]
      
2. xy_overlap=(0.75, 0.75)

![alt text][image4]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?


![alt text][image5]

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:



---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./base_v19.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:



### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

