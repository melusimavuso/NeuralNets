#include "act_defs.h"

// mat mat_fun(const mat& x, double (*fun_p)(double)){
// 	mat M(x.rows(),x.cols());
// 	for (int i=0;i<x.rows();i++){
// 		for (int j=0;j<x.cols();j++){
// 			M(i,j)=fun_p(x(i,j));
// 		}
// 	}
// 	return M;
// }

double relu(double x){
	return std::fmin(std::fmax(x,0),1);
}

double relu_prime(double x){
	if (x>0 and x<1){
		return 1;
	}
	return 0;
}

double sigmoid(double x){
	return 1/(1+std::exp(-1*x));
}

double sigmoid_prime(double x){
	double a = sigmoid(x);
	return a*(1-a);
}

double gaussian(double x){
	return std::exp(-1*x*x);
}

double gaussian_prime(double x){
	return -2*x*gaussian(x);
}

double peceptron(double x){
	if (x>0){
		return 1;
	}
	return 0;
}

double peceptron_prime(double x){
	if(x==0){
		return 1;
	}
	return 0;
}

// double tanh(double x){
// 	return std::tanh(x);
// }

double tanh_prime(double x){
	return 1-std::pow(std::tanh(x),2);
}

double arctan_prime(double x){
	return 1/(1+x*x);
}