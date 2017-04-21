
**Advanced Lane Finding Project**




[//]: # (Image References)

[image1]: ./examples/download.png "Undistorted"
[image2]: ./examples/undistortion.png "Road Transformed"
[image3]: ./examples/binary.png "Binary Example"
[image4]: ./examples/warp.png "Warp Example"
[image5]: ./examples/lane.png "Fit Visual"
[image6]: ./examples/output.png "Output"
[video1]: ./project_output_colour.mp4 "Video"

  

---
**Camera Calibration**

 The code for this step is contained from the second to the 4th code cell of the IPython notebook, p4_submit.ipynb 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

---
**Provide an example of a distortion-corrected image**
 The code for this step is contained from the second to the 6th code cell of the IPython notebook, p4_submit.ipynb 
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

---
**create a thresholded binary image**


I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 5 through 74 in `utils.py`).  Here's an example of my output for this step.  

![alt text][image3]

---
**perspective transform**

The code for this step is contained from the 10th to the 11th code cell of the IPython notebook, p4_submit.ipynb 

The code takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32([
    [120, image.shape[0]],
    [image.shape[1]/2 - 60, image.shape[0]-image.shape[0]/2+110],
    [image.shape[1]/2 + 60, image.shape[0]-image.shape[0]/2+110],
    [image.shape[1] - 120, image.shape[0]]
    ])


dst = np.float32([
    [150,image.shape[0]],
     [150 + 80,0],
     [image.shape[1] - 150 - 80,0],
     [image.shape[1] - 150,image.shape[0]]
    ])    

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 120, 720      | 150, 720      | 
| 580, 470      | 230, 0        |
| 700, 470      | 1050, 0       |
| 1160, 720     | 1130, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_output_colour.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

