#include "PID.h"
#include <uWS/uWS.h>

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {

	  // Twiddle: choose optimal parameters
    this->Kp = Kp;
    this->Ki = Ki;
    this->Kd = Kd;
    sum_cte = 0;
    prev_cte = 0;
}

void PID::UpdateError(double cte) {

	sum_cte += cte;
    p_error = - Kp * cte;
    i_error = - Ki * sum_cte;
    d_error = - Kd * (cte - prev_cte);
    prev_cte = cte;
    //std::cout << "updated error" << std::endl;
}

double PID::TotalError() {

	 return p_error + i_error + d_error;
}

void PID::Restart(uWS::WebSocket<uWS::SERVER> ws){
  std::string reset_msg = "42[\"reset\",{}]";
  ws.send(reset_msg.data(), reset_msg.length(), uWS::OpCode::TEXT);
}
