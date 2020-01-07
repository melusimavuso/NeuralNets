#pragma once

#include"LayerCake.h"


vec silce_vec(const mat_vec& vm, int i, int j);

class ANN
{
private:
  int input_dim;
  int output_dim;
  std::vector<LayerCake> layers = {};

public:
  ANN();
  ANN(int input_dim_, int output_dim_, std::vector<int> Neurons,std::shared_ptr<ActivationFunction> activation,std::shared_ptr<ActivationFunction> out_activation);
  ANN(int input_dim_, int output_dim_,std::vector<int> Neurons, std::string activation_symbol);
  //ANN(int input_dim_, int output_dim_, std::vector<int> Neurons,std::shared_ptr<ActivationFunction> activation);

  void add_layer(int input_dim_, int output_dim_, std::shared_ptr<ActivationFunction> activation);
  void add_layer(int input_dim_, int output_dim_, std::string activation_symbol);
  vec operator()(const vec& x);
  vec call(const vec& x);
  std::vector<LayerCake>& get_layers();
  mat dNN_dx(const vec& x);
  mat_vec d2NN_dx2(const vec& x);
  
  mat_vec dNN_dparams(const vec& x);
  //mat_vec d2NN_dxdp(const vec& x);
  
  std::vector<vec> get_weights();
  void set_weights(const std::vector<vec>& list_of_weights);
 
};