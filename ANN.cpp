#include "ANN.h"

vec silce_vec(const mat_vec& vm, int i, int j){
  int N = vm.size();
  vec V(N);
  for (int k = 0;k<N;k++){
    if (i>=vm[0].rows() or j>=vm[0].cols() or i<0 or j<0){
      std::cout<<"You're screwed!"<<std::endl;
    }
    V(k) = vm[k](i,j);
  }
  return V;
}

ANN::ANN(){};
ANN::ANN (int input_dim_, int output_dim_, std::vector<int> Neurons,std::shared_ptr<ActivationFunction> activation,std::shared_ptr<ActivationFunction> out_activation=std::make_shared<ActivationFunction>()){
if (Neurons.empty()){
  this->layers = {LayerCake(input_dim_,output_dim_,out_activation)};
}
else {
  this->layers.push_back(LayerCake(input_dim_,Neurons[0],activation));

  for (unsigned n = 0; n<Neurons.size()-1;++n){
    this->layers.push_back(LayerCake(Neurons[n],Neurons[n+1],activation));
  }
  this->layers.push_back(LayerCake(Neurons.back(),output_dim_,out_activation));
}
this->input_dim = input_dim_;
this->output_dim = output_dim_;

}

ANN::ANN(int input_dim_, int output_dim_,std::vector<int> Neurons, std::string activation_symbol) {

std::shared_ptr<ActivationFunction> act = sym_to_act(activation_symbol);

std::shared_ptr<ActivationFunction> out_act = std::make_shared<ActivationFunction>();

if (Neurons.empty()) {
this->layers = { LayerCake(input_dim_,output_dim_,out_act) };
}
else {
this->layers.push_back(LayerCake(input_dim_, Neurons[0], act));

for (unsigned n = 0; n < Neurons.size() - 1; ++n) {
	this->layers.push_back(LayerCake(Neurons[n], Neurons[n + 1], act));
}
this->layers.push_back(LayerCake(Neurons.back(), output_dim_, out_act));
}
this->input_dim = input_dim_;
this->output_dim = output_dim_;
}

void ANN::add_layer(int input_dim_, int output_dim_, std::shared_ptr<ActivationFunction> activation){
if (this->layers.empty()){
  this->input_dim = input_dim_;
  this->layers.push_back(LayerCake(input_dim_,output_dim_,activation));
  this->output_dim = output_dim_;
}
else{
auto back_layer = this->layers.back();
this->layers.push_back(LayerCake(back_layer.get_output_dim(),output_dim_,activation));
this->output_dim = output_dim_;
}
}

void ANN::add_layer(int input_dim_, int output_dim_, std::string activation_symbol){
  if (this->layers.empty()){
    this->input_dim = input_dim_;
    this->layers.push_back(LayerCake(input_dim_,output_dim_,activation_symbol));
    this->output_dim = output_dim_;
  }
  else{
  auto back_layer = this->layers.back();
  this->layers.push_back(LayerCake(back_layer.get_output_dim(),output_dim_,activation_symbol));
  this->output_dim = output_dim_;
}
}

vec ANN::operator()(const vec& x){
  vec y = x;
  for (auto& layer: this->layers){
    //std::cout<<"Ran"<<std::endl;
     y = layer(y);
   }
  return y;
}

vec ANN::call(const vec& x){

  return (*this)(x);
}

mat ANN::dNN_dx(const vec& x){
  auto y = this->layers[0](x);
  //my_cube dL_dx();
  mat deriv = this->layers[0].dL_dx(x);
  //std::cout<<deriv.n_rows<<","<<deriv.n_cols<<std::endl;
  for (auto it = this->layers.begin()+1;it!=this->layers.end();it++)
  {

    mat dL_dx = (it->dL_dx)(y);
    deriv = dL_dx*deriv;
    y = (*it)(y);
  }
  return deriv;
}


mat_vec ANN::d2NN_dx2(const vec& x){
//int d = this->layers[0].get_output_dim();
int in_d = this->input_dim;
mat M(in_d,in_d);
vec y = x;
mat_vec sec_der(in_d);
for (int k = 0;k<in_d;k++){
  sec_der[k] = mat::Zero(in_d,in_d);
}
mat fir_der = mat::Identity(in_d,in_d);

for(auto& L:this->layers){
  int out_d = L.get_output_dim();
  mat LayerGrad = L.dL_dx(y);
  mat_vec temp_sec_der(L.get_output_dim());
  mat_vec dL2 = L.d2L_dx2(y);
  for (int k=0;k<out_d;k++){
    
    for (int i = 0;i<in_d;i++){
      auto fir_der_col_i = fir_der.col(i);
      for (int j=0;j<in_d;j++){
        auto fir_der_col_j = fir_der.col(j);
        vec sliced_2nd_der = silce_vec(sec_der,i,j);
        M(i,j) = fir_der_col_i.dot(dL2[k]*fir_der_col_j)+ LayerGrad.row(k).dot(sliced_2nd_der);
        temp_sec_der[k] = M;
      }
      
    }
  }
  y = L(y);
  fir_der = LayerGrad;
  sec_der = temp_sec_der;
}

return sec_der;
}



// std::vector<mat> ANN::dNN_dparams(const mat& x){

  // std::vector<mat> dldp;
  // std::vector<mat> dNN_dp = {};
  // std::vector<mat> temp_dNN_dp = {};
  // auto y = x;
  // for (auto& layer:this->layers){
  //   mat layer_der = layer.dL_dx(y);
  //   //int k = 0;
  //   for (auto& v : dNN_dp){
  //     temp_dNN_dp.push_back(layer_der*v);
  //     //k++;
  //   }

  //   dNN_dp = std::move(temp_dNN_dp);
  //   temp_dNN_dp = {};
  //   dldp = layer.dL_dparams(y);
  //   std::move(dldp.begin(),dldp.end(),std::back_inserter(dNN_dp));
  //   y = layer(y);
  // }

  // return dNN_dp;
// }

mat_vec ANN::dNN_dparams(const vec& x){
  mat_vec dldp;
  mat_vec dNN_dp = {};
  mat_vec temp_dNN_dp = {};
  auto y = x;
  for (auto& layer:this->layers){
    mat layer_der = layer.dL_dx(y);
    //int k = 0;
    for (auto& v : dNN_dp){
      temp_dNN_dp.push_back(layer_der*v);
      //k++;
    }

    dNN_dp = std::move(temp_dNN_dp);
    temp_dNN_dp = {};
    dldp = layer.dL_dparams(y);
    std::move(dldp.begin(),dldp.end(),std::back_inserter(dNN_dp));
    y = layer(y);
  }

  return dNN_dp;
}
// my_cube d2NN_dp2(const vec& x);

// mat_vec d2NN_dxdp(const vec& x){
  
// }

std::vector<LayerCake>& ANN::get_layers(){
	return this->layers;
}

vec_vec ANN::get_weights(){
  std::vector<vec> the_weights = {};
  for (auto& layer:this->layers){
    auto layer_weights = layer.get_weights();
    std::copy(layer_weights.begin(),layer_weights.end(),std::back_inserter(the_weights));
  }
  return the_weights;

}

void ANN::set_weights(const std::vector<vec>& list_of_weights){
  auto start_ptr = list_of_weights.begin();
  for (auto& layer:this->layers){
    auto end_ptr = start_ptr+layer.get_output_dim();
    layer.set_weights(std::vector<vec>(start_ptr,end_ptr));
    start_ptr = end_ptr;
  }

}