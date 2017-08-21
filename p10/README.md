[//]: # (Image References)
[image0]: ./resources/mpc_model.jpg 

## Basic Build Instructions

This repo contains 10 projects and too large to clone. Alternatively, p10.zip file can be downloaded. (p10.zip contains the project file.)

1. Clone this repo.
2. Make a build directory: mkdir build && cd build
3. Compile: cmake .. && make
4. Run it: ./mpc


## Result

The vehicle drove on the line before the steep curve just before and after the bridge. Except that part, the vehicle drove on the track safely, driving inside lane. test.mp4 was the test result.


## The model

The MPC model was made based on udacity class.

![alt text][image0]

## Time Length and Elapsed Duration (N & dt)

I used below parameter values. 

N and dt were decided how well MPC line was drawn. 
```
size_t N = 12;
double dt = 0.05;
```

## Polynominal Fitting and MPC Preprocessing

1. projected x,y coordinate was transformed using below equation. 
( https://discussions.udacity.com/t/mpc-car-space-conversion-and-output-of-solve-intuition/249469/12 )



```
Eigen::MatrixXd transformGlobalToLocal(double x, double y, double psi, const vector<double> & ptsx, const vector<double> & ptsy) {

    assert(ptsx.size() == ptsy.size());
    unsigned len = ptsx.size();

    auto waypoints = Eigen::MatrixXd(2,len);

    for (auto i=0; i<len ; ++i){
      waypoints(0,i) =   cos(psi) * (ptsx[i] - x) + sin(psi) * (ptsy[i] - y);
      waypoints(1,i) =  -sin(psi) * (ptsx[i] - x) + cos(psi) * (ptsy[i] - y);  
    } 

    return waypoints;

}
```

2. CTE and Epsi were calculated before feeding the data to MPC processing.


## Model Predictive Control with Latency

My reviewer suggests...

```
Two common aproaches exist to take delays into account:

In one approach the prospective position of the car is estimated based on its current speed and heading
direction by propagating the position of the car forward until the expected time when actuations are
expected to have an effect. The NMPC trajectory is then determined by solving the control problem starting
from that position.
In the other approach the control problem is solved from the current position and time onwards. Latency is
taken into account by constraining the controls to the values of the previous iteration for the duration
of the latency. Thus the optimal trajectory is computed starting from the time after the latency period.
This has the advantage that the dynamics during the latency period is still calculated according to the
vehicle model.
```

I did the second approach and 0.1 Latency was used.

```
const int latency_ind = 2;
dt = 0.05

0.1 = 2 * 0.05

```

Atfter cost was minimized on the logic, steer_value & throttle_value were fed into the simulator. 

```
 
 double steer_value = sol.Delta.at(latency_ind);
 double throttle_value= sol.A.at(latency_ind);

 ```
 151 and 152 line in the file, main.cpp. 



Latency, 100 ms, were also used on the model.

```
this_thread::sleep_for(chrono::milliseconds(100));
```