#include "Activations.cpp"
#include "LayerCake.cpp"
#include "ANN.cpp"
#include "act_defs.cpp"
#include "single_layer_nn.cpp"

using namespace std;

template <class T>
void print_vec(std::vector<T> vvec){
	for (auto& v:vvec){
		cout<<v<<endl;
		cout<<".....------......"<<endl;
	}
}

void NeuronTesting(){
	int in_dim = 2;
	Neuron n(in_dim, std::make_shared<sin_act>());
	vec x = vec::Random(in_dim);
	auto y = n(x);
	cout<<"x is "<<endl<<x<<endl;
	cout<<"y is "<<endl<<y<<endl;
	vec der = n.dn_dx(x);
	cout<<"Now dn_dx = "<<endl;
	cout<<der<<endl;
	auto sec_der = n.d2n_dx2(x);
	cout<<"d2n_dx2 = "<<endl;
	cout<<sec_der<<endl;
	auto dn_dp = n.dn_dp(x);
	cout<<"dn_dp is "<<endl<<dn_dp<<endl;
	auto d2n_dp2 = n.d2n_dp2(x);
	cout<<"d2n_dp2 is "<<endl<<d2n_dp2<<endl;
	auto d2n_dxdp = n.d2n_dxdp(x);
	cout<<"d2n_dxdp is "<<endl<<d2n_dxdp<<endl;
	auto weights = n.get_weights();
	cout<<"weights = "<<endl<<weights<<endl;
	n.set_weights(weights);
	auto weights2 = n.get_weights();
	cout<<"weights2 = "<<endl<<weights2<<endl;
}

void LayerTesting(){
	int in_dim = 2;
	int out_dim = 4;
	LayerCake L(in_dim, out_dim, std::make_shared<sin_act>());
	vec x = vec::Random(in_dim);
	auto y = L(x);
	cout<<"x is "<<endl<<x<<endl;
	cout<<"y is "<<endl<<y<<endl;

	auto der = L.dL_dx(x);
	cout<<"Now dL_dx = "<<endl;
	cout<<der<<endl;

	auto sec_der = L.d2L_dx2(x);
	cout<<"d2L_dx2 = "<<endl;
	print_vec(sec_der);

	auto dL_dp = L.dL_dparams(x);
	cout<<"dL_dp is "<<endl;
	print_vec(dL_dp);

	auto d2L_dp2 = L.d2L_dp2(x);
	cout<<"d2L_dp2 is "<<endl;
	print_vec(d2L_dp2);

	// auto d2L_dxdp = L.d2L_dxdp(x);
	// cout<<"d2L_dxdp is "<<endl;
	// print_vec(d2L_dxdp);

	auto weights = L.get_weights();
	cout<<"weights = "<<endl;
	print_vec(weights);

	L.set_weights(weights);
	
	auto weights2 = L.get_weights();
	cout<<"weights2 = "<<endl;
	print_vec(weights2);
}

void ANNTesting(){
	int in_dim = 2;
	int out_dim = 4;
	std::vector<int> Neurs = {4,3};
	ANN N;
	N.add_layer(2,4,"sin");
	N.add_layer(4,3,"sin");
	N.add_layer(3,4,"lin");
	vec x = vec::Random(in_dim);
	auto y = N(x);
	cout<<"x is "<<endl<<x<<endl;
	cout<<"y is "<<endl<<y<<endl;

	auto der = N.dNN_dx(x);
	cout<<"Now dNN_dx = "<<endl;
	cout<<der<<endl;

	auto sec_der = N.d2NN_dx2(x);
	cout<<"d2NN_dx2 = "<<endl;
	print_vec(sec_der);

	auto dN_dp = N.dNN_dparams(x);
	cout<<"dN_dp is "<<endl;
	print_vec(dN_dp);
	
	auto weights = N.get_weights();
	cout<<"weights = "<<endl;
	print_vec(weights);
	N.set_weights(weights);
	auto weights2 = N.get_weights();
	cout<<"weights2 = "<<endl;
	print_vec(weights2);
}

void slnnTesting(){
	int in_dim = 2;
	int num_neurons = 10;
	std::string act_sym = "sigmoid";
	single_layer_nn N(in_dim,num_neurons,act_sym);
	vec x = vec::Random(in_dim);
	auto y = N(x);
	cout<<"x is "<<endl<<x<<endl;
	cout<<"y is "<<endl<<y<<endl;

	auto der = N.dN_dx(x);
	cout<<"Now dNN_dx = "<<endl;
	cout<<der<<endl;

	auto sec_der = N.d2N_dx2(x);
	cout<<"d2NN_dx2 = "<<endl;
	cout<<sec_der<<endl;

	auto dN_dp = N.dN_dp(x);
	cout<<"dN_dp is "<<endl;
	print_vec(dN_dp);
	
	auto weights = N.get_weights();
	cout<<"weights = "<<endl;
	print_vec(weights);
	N.set_weights(weights);
	auto weights2 = N.get_weights();
	cout<<"weights2 = "<<endl;
	print_vec(weights2);
};