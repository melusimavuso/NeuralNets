#pragma once
#include<vector>
#include <memory>
#include"Activations.h"

mat append_one(const mat& A);

class Neuron
{
	int input_dim;
	vec weights;
	std::shared_ptr<ActivationFunction> act;
public:	
	Neuron();
	Neuron(int input_dim_);
	Neuron(int input_dim_, std::shared_ptr<ActivationFunction> act);

	double operator()(const vec& x);

	vec dn_dx(const vec& x);
	mat d2n_dx2(const vec& x);

	vec dn_dp(const vec& x);
	mat d2n_dp2(const vec& x);

	mat d2n_dxdp(const vec& x);

	vec& get_weights();
	void set_weights(const vec& list_of_weights);	
};

class LayerCake {
private:
	int input_dim;
	int output_dim;
	std::vector<Neuron> neurons;
	std::shared_ptr<ActivationFunction> act;

public:
	LayerCake();
	LayerCake(int input_dim_, int output_dim_);
	LayerCake(int input_dim_, int output_dim_, std::shared_ptr<ActivationFunction> activation);
	LayerCake(int input_dim_, int output_dim_, std::string activation_symbol);

	int& get_output_dim();
	int& get_input_dim();

	vec operator()(const vec& x);
	vec call(const vec& x);

	mat dL_dx(const vec& x);
	mat_vec d2L_dx2(const vec& x);
	
	mat_vec dL_dparams(const vec& x);
	mat_vec d2L_dp2(const vec& x);

	//std::vector<mat_vec> d2L_dxdp(const vec& x);

	vec_vec get_weights();
	void set_weights(const vec_vec& list_of_weights);
};

