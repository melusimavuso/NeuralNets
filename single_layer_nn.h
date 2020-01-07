#pragma once
#include "ANN.h"

class single_layer_nn
{
public:
	int input_dim;
	int num_neurons;
	ANN NN;
	std::shared_ptr<ActivationFunction> act;

	single_layer_nn();
	single_layer_nn(int input_dim_, int num_neurons);
	single_layer_nn(int input_dim_, int num_neurons, std::string activation_symbol);

	double operator()(const vec& x);
	double call(const vec& x);

	vec dN_dx(const vec& x);
	mat d2N_dx2(const vec& x);

	vec_vec dN_dp(const vec& x);

	mat_vec d2N_dxdp(const vec& x);
	// std::vector<mat_vec> d3N_dx2dp(const vec& x);

	vec_vec get_weights();
	void set_weights(const vec_vec& list_of_weights);
	
};