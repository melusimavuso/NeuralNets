#include "single_layer_nn.h"

single_layer_nn::single_layer_nn(){
	this->input_dim = 1;
	this->num_neurons = 0;
	this->act = std::shared_ptr<ActivationFunction>();
}

single_layer_nn::single_layer_nn(int input_dim_, int num_neurons_){
	this->input_dim = input_dim_;
	this->num_neurons = num_neurons_;
	std::vector<int> neurs = {num_neurons_};
	this->NN = ANN(input_dim_,1, neurs,std::shared_ptr<ActivationFunction>());
}

single_layer_nn::single_layer_nn(int input_dim_, int num_neurons_, std::string activation_symbol){
	this->input_dim = input_dim_;
	this->num_neurons = num_neurons_;
	std::vector<int> neurs = {num_neurons_};
	this->NN = ANN(input_dim_,1, neurs, activation_symbol);
	this->act = sym_to_act(activation_symbol);
}

double single_layer_nn::operator()(const vec& x){
	return NN(x)(0);
}

double single_layer_nn::call(const vec& x){
	return (*this)(x);
}

vec single_layer_nn::dN_dx(const vec& x){
	vec ans = NN.dNN_dx(x).row(0);
	return ans;
}

mat single_layer_nn::d2N_dx2(const vec& x){
	mat ans = NN.d2NN_dx2(x)[0];
	return ans;
}

vec_vec single_layer_nn::dN_dp(const vec& x){
	vec_vec ans = {};
	auto dndp = NN.dNN_dparams(x);
	for (auto& der: dndp){
		vec v = der.row(0);
		ans.push_back(v);
	}
	return ans;
}

mat_vec single_layer_nn::d2N_dxdp(const vec& x){
	return mat_vec();
}

// std::vector<mat_vec> single_layer_nn::d3N_dx2dp(const vec& x);

vec_vec single_layer_nn::get_weights(){
	return NN.get_weights();
}

void single_layer_nn::set_weights(const vec_vec& list_of_weights){
	NN.set_weights(list_of_weights);
}