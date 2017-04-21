
**Advanced Lane Finding Project**




[//]: # (Image References)

[image1]: ./examples/download.png "Undistorted"
[image2]: ./examples/undistortion.png "Road Transformed"
[image3]: ./examples/binary.png "Binary Example"
[image4]: ./examples/warp.png "Warp Example"
[image5]: ./examples/lane.png "Fit Visual"
[image6]: ./examples/output.png "Output"
[image7]: ./examples/failure.jpg "Output"
[video1]: ./project_output_colour.mp4 "Video"

  

---
**Camera Calibration**

 The code for this step is contained from the second to the 4th code cell of the IPython notebook, p4_submit.ipynb 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

---
**an example of a distortion-corrected image**

 The code for this step is contained from the second to the 6th code cell of the IPython notebook, p4_submit.ipynb. 

![alt text][image2]

---
**thresholded binary image**


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

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image. I just select center triangle area of intereset for perspective transform. 

![alt text][image4]

---
**fit lane lines**

To fit my lane lines with a 2nd order polynomial fit,  I did some other stuff.
(steps at lines 77 through 249 in utils.py)

- I made search window in vertical direction, devided by 6.
- find multiple vetical peaks in left and right side of the transformed image
- search more pixels based on these peaks within 200 pixels, window radius.

![alt text][image5]


reference: https://github.com/jessicayung/self-driving-car-nd/tree/master/p4-advanced-lane-lines

---
**the radius of curvature of the lane and the position of the vehicle with respect to center**

The code for this step is contained from the 17th to the 19th code cell of the IPython notebook, p4_submit.ipynb. 
I followed the code which udacity provides to calculate curvature in each direction and did addtional calculation.

- curvature = (left_curverad + right_curverad) / 2
- center = (1.5 * left_second_order_poly - right_second_order_poly) / 2

---
**example image**

![alt text][image6]

---

**Pipeline (video)**

Here's a [link to my video result](./project_output_colour.mp4)

---

**Discussion**

I faced failure to draw proper lane lines on the frame no 1001 and 1002 caused from lack of right lanes on the bottom of image.
To make my lane lines more robust, I simply calculated (right_bottom pixel - left_bottom pixel) and dropped lane lines below 700 pixels.  

The code for this step is contained in the 22th code cell of the IPython notebook, p4_submit.ipynb 

```
- bottom = fit[0] * 720 ** 2 +  fit[1] * 720 + fit[2]
-  if((right_bottom - left_bottom) < 700):
        left_coeffs = prev_left_coeffs
        right_coeffs = prev_right_coeffs

```

![alt text][image7]

It works! However there are other ways to implement more robust solutions.  

