#include"Activations.h"


ActivationFunction::ActivationFunction() {};

std::shared_ptr<ActivationFunction> sym_to_act(std::string act_sym){
	std::shared_ptr<ActivationFunction> act;

    if (act_sym=="cos")
    {
    	act = std::make_shared<cos_act>();
    }

    else if (act_sym=="sin")
    {
    	act = std::make_shared<sin_act>();
    }

    else if (act_sym=="lin")
    {
    	act = std::make_shared<ActivationFunction>();
    }

    else if (act_sym=="relu")
    {
    	act = std::make_shared<relu_act>();
    }

    else if (act_sym=="sigmoid")
    {
    	act = std::make_shared<sigmoid_act>();
    }

    else if (act_sym=="gaussian")
    {
    	act = std::make_shared<gaussian_act>();
    }

    else if (act_sym=="perceptron")
    {
    	act = std::make_shared<peceptron_act>();
    }

    else if (act_sym=="tanh")
    {
    	act = std::make_shared<tanh_act>();
    }

    else if (act_sym=="arctan")
    {
    	act = std::make_shared<arctan_act>();
    }

    else
    {
    	act = std::make_shared<ActivationFunction>();
    }

    return act;
}


double ActivationFunction::f(double x) {

	return x;
}
double ActivationFunction::f_prime(double x) {
	//x.ones();
	return 1;
	}

double ActivationFunction::f_d_prime(double x) {
    //x.ones();
    return 0;
    }

//std::map<key, value> map;

double sin_act::f(double x) {
	return std::sin(x);
}
double sin_act::f_prime(double x) {
	return std::cos(x);
}

double sin_act::f_d_prime(double x) {
    return -1*std::sin(x);
}

double cos_act::f(double x) {
	return std::cos(x);
}
double cos_act::f_prime(double x) {
	return -1*std::sin(x);
}

double cos_act::f_d_prime(double x) {
    return -1*std::cos(x);
}

double relu_act::f(double x){
	return relu(x);
}

double relu_act::f_prime(double x) {
	return relu_prime(x);
}

double relu_act::f_d_prime(double x) {
    return 0;
}

double sigmoid_act::f(double x){
	return sigmoid(x);
}

double sigmoid_act::f_prime(double x) {
	return sigmoid_prime(x);
}

double sigmoid_act::f_d_prime(double x) {
    return sigmoid_act::f_prime(x)*(1-2*sigmoid_act::f(x));
}

double gaussian_act::f(double x){
	return gaussian(x);
}

double gaussian_act::f_prime(double x) {
	return gaussian_prime(x);
}

double gaussian_act::f_d_prime(double x) {
    double n_x_sq = -1*x*x;

    return -2*std::exp(n_x_sq)*(1-n_x_sq);
}

double peceptron_act::f(double x){
	return peceptron(x);
}

double peceptron_act::f_prime(double x) {
	return peceptron_prime(x);
}

double peceptron_act::f_d_prime(double x) {
    return 0;
}


double tanh_act::f(double x){
	return std::tanh(x);
}

double tanh_act::f_prime(double x) {
	return tanh_prime(x);
}

double tanh_act::f_d_prime(double x) {
    return -2*tanh_act::f(x)*tanh_act::f_prime(x);
}

double arctan_act::f(double x){
	return std::atan(x);
}

double arctan_act::f_prime(double x) {
	return arctan_prime(x);
}

double arctan_act::f_d_prime(double x) {
    return 2*x*pow(arctan_act::f_prime(x),2);
}



