This project fuses the position and velocity measurements of obstacles from rader and lasar measurements to track the obstacles through time.  

[//]: # (Image References)

[image0]: ./resources/ukf_test.jpg 
[image1]: ./resources/ukf_equation.jpg 
[image2]: ./resources/sv.jpg 
[image3]: ./resources/sp_gen.jpg 


## Basic Build Instructions

This repo contains 10 projects and too large to clone. Alternatively, p7.zip file can be downloaded. (p7.zip contains the project file.)

1. Clone this repo.
2. Make a build directory: mkdir build && cd build
3. Compile: cmake .. && make
4. Run it: ./UnscentedKF


## Accuracy

The Kalman Filter was able to track obstacles accuractely and better than previous extended kalman filter.

![alt text][image0]

## Follows the Correct Algorithm

The transition and measurement functions are nonlinear, but rather than a Taylor approximation used in Extended Kalman Filter, it uses a different approximation scheme called the unscented transform and then feed the results into a Kalman filter.

In that point, UKF implementation is different from EKF implementation. 

1. UKF is CRTV model, constant turn rate and velocity magnitude model (CTRV).

![alt text][image2]

2. UKF generates sigma points and calculate predicted mean and covaraince of sigma points during prediction step. 

 ![alt text][image3]

3. UKF Cross-Corelation Matrix during update step

 ![alt text][image1]

 ## Summary 

 UKF is more accurate than EKF and uses more resources than EKF. 

