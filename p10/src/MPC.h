#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

#define DT 0.1// time step duration dt in s 

//the distance between the front of the vehicle and its center of gravity
#define LF 2.67 

using namespace std;

const int latency_ind = 2;


struct Solution {

		vector<double> X;
		vector<double> Y;
		vector<double> Psi;
		vector<double> V;
		vector<double> Cte;
		vector<double> Epsi;
		vector<double> Delta;
		vector<double> A;

};

class MPC {
 public:
  MPC();

  virtual ~MPC();

  // Solve the model given an initial state and polynomial coefficients.
  // Return the first actuatotions.
  //vector<double> Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);
  Solution Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);


  // previoud Delta and acceleration/decceleration
  double prevDelta {0};
  double prevA {0.1};
};

#endif /* MPC_H */
