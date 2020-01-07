#pragma once

#include <iostream>
#include "Trivial.cpp"
#include <string>
#include "act_defs.h"


//mat mat_fun(const mat& x, double (*fun_p)(double));

class ActivationFunction {
public:
	ActivationFunction();
	virtual double f(double x);
	virtual double f_prime(double x);
	virtual double f_d_prime(double x);
};

class sin_act :public ActivationFunction {
public:
	double f(double x);
	double f_prime(double x);
	double f_d_prime(double x);
};

class cos_act :public ActivationFunction {
public:
	double f(double x);
	double f_prime(double x);
	double f_d_prime(double x);
};

class relu_act: public ActivationFunction {
public:
	double f(double x);
	double f_prime(double x);
	double f_d_prime(double x);
};

class sigmoid_act: public ActivationFunction {
public:
	double f(double x);
	double f_prime(double x);
	double f_d_prime(double x);
};

class gaussian_act: public ActivationFunction {
public:
	double f(double x);
	double f_prime(double x);
	double f_d_prime(double x);
};

class peceptron_act: public ActivationFunction {
public:
	double f(double x);
	double f_prime(double x);
	double f_d_prime(double x);
};

class tanh_act: public ActivationFunction {
public:
	double f(double x);
	double f_prime(double x);
	double f_d_prime(double x);
};

class arctan_act: public ActivationFunction {
public:
	double f(double x);
	double f_prime(double x);
	double f_d_prime(double x);
};

// class leaky_relu_act: public ActivationFunction {
// public:
// 	virtual mat f(mat x);
// 	virtual mat f_prime(mat x);
// };

// class elu_act: public ActivationFunction {
// public:
// 	virtual mat f(mat x);
// 	virtual mat f_prime(mat x);
// };

// class softmax_act: public ActivationFunction {
// public:
// 	virtual mat f(mat x);
// 	virtual mat f_prime(mat x);
// };

std::shared_ptr<ActivationFunction> sym_to_act(std::string act_sym);