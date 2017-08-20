[//]: # (Image References)
[image0]: ./resources/mpc_model.jpg 

## Basic Build Instructions

This repo contains 10 projects and too large to clone. Alternatively, p10.zip file can be downloaded. (p10.zip contains the project file.)

1. Clone this repo.
2. Make a build directory: mkdir build && cd build
3. Compile: cmake .. && make
4. Run it: ./mpc


## Result

The vehicle drove on the line before the steep curve just after the bridge. Except that part, the vehicle
drove on the track safely that drove inside lane. test.mp4 was the test result.


## The model

The MPC model was made based on udacity class.

![alt text][image0]

## Time Length and Elapsed Duration (N & dt)

I used below parameter values. 

N was decided how well MPC line was drawn. 12 was better than 10 in my case.
dt value followed udacity discussion forum. 
( https://discussions.udacity.com/t/need-help-in-implementing-mpc-project/257510/4 )

size_t N = 12;
double dt = 0.1;


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

Latency, 100 ms, were used on the model.

```
this_thread::sleep_for(chrono::milliseconds(100));
```