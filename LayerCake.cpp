#include "LayerCake.h"

vec append_one(const vec& A){
	vec B(A.size()+1);
	vec aa(1);
	aa<<1;
	B<<A,aa;

	return B;
}

Neuron::Neuron(){
	this->input_dim = 1;
	int dim_ = this->input_dim + 1;
	this->weights = vec::Random(dim_ );
	this->act = std::make_shared<ActivationFunction>();
}

Neuron::Neuron(int input_dim_, std::shared_ptr<ActivationFunction> act_){
	this->input_dim = input_dim_;
	int dim_ = this->input_dim + 1;
	this->weights = vec::Random(dim_ );
	this->act = act_;
}

Neuron::Neuron(int input_dim_){
	this->input_dim = input_dim_;
	int dim_ = this->input_dim + 1;
	this->weights = vec::Random(dim_ );
	this->act = std::make_shared<ActivationFunction>();
}

double Neuron::operator()(const vec& x){
	return (this->act)->f(this->weights.dot(append_one(x)));
}

vec Neuron::dn_dx(const vec& x){
	double out_prime = (this->act)->f_prime(this->weights.dot(append_one(x)));
	int in_dim = this->input_dim;
	return out_prime*((this->weights).head(in_dim));
}

mat Neuron::d2n_dx2(const vec& x){
	double out_d_prime = (this->act)->f_d_prime(this->weights.dot(append_one(x)));
	mat M(this->input_dim,this->input_dim);
	for (int i=0;i<this->input_dim;i++){
		for (int j=0;j<this->input_dim;j++){
			M(i,j) = out_d_prime*(this->weights(i))*(this->weights(j));
		}
	}

	return M;
}

vec Neuron::dn_dp(const vec& x){
	vec x1 = append_one(x);
	double out_prime = (this->act)->f_prime(this->weights.dot(append_one(x)));
	return out_prime*x1;

}

mat Neuron::d2n_dp2(const vec& x){
	vec x1 = append_one(x);
	double out_d_prime = (this->act)->f_d_prime(this->weights.dot(x1));
	mat M(this->input_dim+1,this->input_dim+1);
	for (int i=0;i<this->input_dim+1;i++){
		for (int j=0;j<this->input_dim+1;j++){
			M(i,j) = out_d_prime*(x1(i))*(x1(j));
		}
	}

	return M;
}

mat Neuron::d2n_dxdp(const vec& x){
	vec x1 = append_one(x);
	vec out_prime_w = (this->act)->f_prime(this->weights.dot(x1))*this->weights;
	double out_d_prime = (this->act)->f_d_prime(this->weights.dot(x1));
	mat M(this->input_dim,this->input_dim+1);
	for (int i=0;i<this->input_dim;i++){
		for (int j=0;j<this->input_dim+1;j++){
			if (j!=i){
				M(i,j) = out_d_prime*this->weights(i)*x1(j);
			}
			else{
				M(i,j) = out_d_prime*this->weights(i)*x1(j)+out_prime_w(i);
			}
		}
	}
	return M;
}


vec& Neuron::get_weights(){
	return this->weights;
}

void Neuron::set_weights(const vec& weights_){
	this->weights = weights_;
}


//Now........................Layers


LayerCake::LayerCake() {
	this->act = std::make_shared<ActivationFunction>();
}
LayerCake::LayerCake(int input_dim_, int output_dim_) {
	this->input_dim = input_dim_;
	this->output_dim = output_dim_;
	auto acti = std::make_shared<ActivationFunction>();
	this->act = acti;
	this->neurons = std::vector<Neuron>(output_dim_);
	for (int k=0;k<output_dim_;k++){
		this->neurons[k] = Neuron(input_dim_,acti);
	}

}

LayerCake::LayerCake(int input_dim_, int output_dim_, std::shared_ptr<ActivationFunction> activation) {
	this->input_dim = input_dim_;
	this->output_dim = output_dim_;
	this->act = activation;
	this->neurons = std::vector<Neuron>(output_dim_);
	for (int k=0;k<output_dim_;k++){
		this->neurons[k] = Neuron(input_dim_, activation);
	}
}

LayerCake::LayerCake(int input_dim_, int output_dim_, std::string activation_symbol) {
	this->input_dim = input_dim_;
	this->output_dim = output_dim_;
	this->neurons = std::vector<Neuron>(output_dim_);
	auto acti  = sym_to_act(activation_symbol);
	for (int k=0;k<output_dim_;k++){
		this->neurons[k] = Neuron(input_dim_,acti);
	}
	this->act = acti;
}

int& LayerCake::get_output_dim() {
	return this->output_dim;
}
int& LayerCake::get_input_dim() {
	return this->input_dim;
}


vec LayerCake::operator()(const vec& x) {
	vec V(this->output_dim);
	for (int k = 0;k<this->output_dim;k++){
		V(k) = this->neurons[k](x);
	}
	return V;
}

vec LayerCake::call(const vec& x){
	return (*this)(x);
}

mat LayerCake::dL_dx(const vec& x) {
	mat M(this->output_dim,this->input_dim);
	for (int i = 0;i<this->output_dim;i++){
		M.row(i) = this->neurons[i].dn_dx(x);
	}
	return M;
}

mat_vec LayerCake::dL_dparams(const vec& x) {
	mat_vec the_derivs = {};
	//vec x1 = append_one(x);
	int d = this->input_dim+1;
	mat M = mat::Zero(this->output_dim,d);
	for (int k = 0;k<this->output_dim;k++){
		M.row(k) = this->neurons[k].dn_dp(x);
		the_derivs.push_back(M);
		M.row(k) = vec::Zero(d);
	}

	return the_derivs;
}
	
mat_vec LayerCake::d2L_dx2(const vec& x){
	mat_vec Hessian(this->output_dim);
	//ma M = mat::Zero(this->input_dim,this->input_dim);
	for (int k = 0;k<this->output_dim;k++){
		Hessian[k] = this->neurons[k].d2n_dx2(x);
	}

	return Hessian;
}

mat_vec LayerCake::d2L_dp2(const vec& x){
	int the_dim = this->output_dim*(this->input_dim+1);
	mat M;
	mat_vec Hessian(this->output_dim);
	int ind = 0;

	for (int k = 0;k<this->output_dim;k++){
		M = mat::Zero(the_dim,the_dim);
		M.block(ind, ind, this->input_dim+1,this->input_dim+1)=this->neurons[k].d2n_dp2(x);
		Hessian[k] = M;
		ind+=(this->input_dim+1);
	}

	return Hessian;
}

// std::vector<mat_vec> LayerCake::d2L_dxdp(const vec& x){
// 	mat_vec M = {};
// 	std::vector<mat_vec> Vecs = {};
// 	int k = 0;
// 	for (auto& n:this->neurons){
// 		M.push_back(n.d2n_dxdp(x));
		
// 	}

// 	return M;
//}

vec_vec LayerCake::get_weights() {
	vec_vec ans;
	for (auto& N:this->neurons){
		ans.push_back(N.get_weights());
	}
	return ans;
}

void LayerCake::set_weights(const vec_vec& list_of_weights) {
	int k=0;
	for (auto& N:this->neurons){
		N.set_weights(list_of_weights[k]);
		k++;
	}
}