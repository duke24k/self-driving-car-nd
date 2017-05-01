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

[image8]: ./examples/svm_para2.png

[video1]: ./base_v19.mp4


---

**I. Histogram of Oriented Gradients (HOG)**

**extracting HOG features from the training images.**

 The code for this step is contained from the second code cell to the 5th cell of the IPython notebook file, svc_train.ipynb. 
 
 I started by reading in all the vehicle and non-vehicle images. I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. Here is an example using the `YCrCb` color space with 1 channel and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Car 
![alt text][image0]


Not Car
![alt text][image1]


**final choice of HOG parameters.**

I tried various combinations of parameters to detect cars on the test images. Here is a example how I tested hog parameters. Trained SVM performance varies depending on parameters and training times. I choosed SVM which could detect cars on test images even though there were some false positives. Here are some example images:

![alt text][image2]


**training a classifier** 

1. Format features using np.vstack and StandardScaler().
2. Split data into shuffled training and test sets
3. Train linear SVM using sklearn.svm.LinearSVC().
 
 The code for this step is contained in the 6th cell of the IPython notebook file, svc_train.ipynb. 

---
**II.Sliding Window Search** 


I decided to search random window positions using below pyramid and finally came up with this 

1. pyramid = [
           ((64, 64),  [400, 500]),
           ((96, 96),  [400, 500]),
           ((128, 128),[450, 578])
      ]
      
2. xy_overlap=(0.75, 0.75)

![alt text][image4]


**optimizing the performance of classifier**

I used 'LinearSVC.decision_function' to optimize the performance of classifier. I tried default cutoff as below.

- cutoff = (np.mean(dicision_values) + np.max(dicision_values)) // 2.1.

After the default value, I adjusted the cutoff value after checking below dicision value distribution graph on the sliding window.

![alt text][image8]


Ultimately I decided to use YCrCb 1-channel HOG features plus 32 spatially binned color and 64 histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]

---
**III. Video Implementation**


Here's a [link to my video result](./base_v19.mp4)


**filter for false positives and some method for combining overlapping bounding boxes.**

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then accumulated that map using 'deque(maxlen=30)' to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here are 9 test image frames and their accumulated heatmaps:

![alt text][image6]


However there were still false positives and I made 

![alt text][image7]
 

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

### Here the resulting bounding boxes are drawn onto the last frame in the series:




---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

