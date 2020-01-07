#include "Activations.cpp"
#include "LayerCake.cpp"
#include "ANN.cpp"
#include "Model.cpp"
#include "Application.cpp"
#include "Forecasting.cpp"
#include "Hedging.cpp"
#include "training.cpp"
#include "act_defs.cpp"
#include "single_layer_nn.cpp"
#include "PDE.cpp"
//#include "training.cpp"

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


void ModelTesting(){
	int x_dim = 2;
	int y_dim = 4;
	int c_dim = 6;
	std::vector<int> gNeurs = {4,3};
	std::vector<int> fNeurs = {4,3};

	NN_Model M(x_dim,y_dim,c_dim,"sigmoid","cos",gNeurs,fNeurs);
	vec x = vec::Random(x_dim);
	vec C = vec::Random(c_dim);
	C = M.g(x,C);
	auto y = M.f(C);
	cout<<"x is "<<endl<<x<<endl;
	cout<<"C is "<<endl<<C<<endl;
	cout<<"y is "<<endl<<y<<endl;

	auto der_g = M.dg_dC(x,C);
	cout<<"Now dg_dC = "<<endl;
	cout<<der_g<<endl;

	auto der_f = M.df_dC(C);
	cout<<"Now df_dC = "<<endl;
	cout<<der_f<<endl;

	auto dg_dparams = M.dg_dparams(x,C);
	cout<<"dg_dparams = "<<endl;
	print_vec(dg_dparams);

	auto df_dparams = M.df_dparams(C);
	cout<<"df_dparams = "<<endl;
	print_vec(df_dparams);
	
	auto weights = M.get_weights();
	cout<<"weights = "<<endl;
	print_vec(weights);
	M.set_weights(weights);
	auto weights2 = M.get_weights();
	cout<<"weights2 = "<<endl;
	print_vec(weights2);
}

void ForecastingTesting(){
	int x_dim = 4;
	int y_dim = 3;
	int c_dim = 6;
	std::vector<int> gNeurs = {4,3};
	std::vector<int> fNeurs = {4,3};

	Forecasting F(x_dim,y_dim,c_dim,"sigmoid","cos",gNeurs,fNeurs);
	
	std::vector<vec_vec> data(5);
	std::vector<vec_vec> actual(5);
	for (int i=0;i<5;i++){
		vec_vec path(10+i);
		vec_vec apath(10+i);
		for (int j = 0;j<10+i;j++){
			vec x = vec::Random(x_dim);
			path[j] = x;
			apath[j] = x.head(y_dim);
		}

		data[i] = path;
		actual[i] = apath;
	}
	cout<<"Let's start"<<endl;
	F.split_data(data);
	double av_err = F.average_error();
	cout<<"The average error is "<<av_err<<endl;

	int p_num = 0;
	auto path_error_grad = F.path_error_grad(p_num);
	cout<<"The path error grad for path "<<p_num<<" is "<<endl;
	print_vec(path_error_grad);

	F.error_grad(2);
	cout<<"The error grad for is "<<endl;
	print_vec(path_error_grad);

	auto pred = F.predict(data[p_num]);
	cout<<"The prediction for path "<<p_num<<" is "<<endl;
	print_vec(pred);
	cout<<"The actual path "<<p_num<<" is "<<endl;
	print_vec(actual[p_num]);

	auto weights = F.get_params();
	cout<<"weights = "<<endl;
	print_vec(weights);
	F.set_params(weights);
	auto weights2 = F.get_params();
	cout<<"weights2 = "<<endl;
	print_vec(weights2);
}

void HedgingTesting(){
	int x_dim = 4;
	int y_dim = 3;
	int c_dim = 6;
	std::vector<int> gNeurs = {4,3};
	std::vector<int> fNeurs = {4,3};

	Hedging H(x_dim,y_dim,c_dim,"sigmoid","cos",gNeurs,fNeurs);
	
	std::vector<vec_vec> data(5);
	for (int i=0;i<5;i++){
		vec_vec path(10+i);
		for (int j = 0;j<10+i;j++){
			vec x = vec::Random(x_dim);
			path[j] = x;
		}

		auto last = path.back();
		vec v(1);
		v(0) = last.squaredNorm();

		path.push_back(v);

		data[i] = path;
	}

	cout<<"Let's start"<<endl;
	H.split_data(data);
	double av_err = H.average_error();
	cout<<"The average error is "<<av_err<<endl;

	int p_num = 0;
	auto path_error_grad = H.path_error_grad(p_num);
	cout<<"The path error grad for path "<<p_num<<" is "<<endl;
	print_vec(path_error_grad);

	H.error_grad(2);
	cout<<"The error grad is "<<endl;
	print_vec(path_error_grad);

	auto pred = H.predict(vector<vec>(data[p_num].begin(),data[p_num].end()-1));
	cout<<"The prediction for path "<<p_num<<" is "<<endl;
	print_vec(pred);

	auto value = H.value(vector<vec>(data[p_num].begin(),data[p_num].end()-1));
	cout<<"The value for path "<<p_num<<" is "<<value<<endl;


	auto weights = H.get_params();
	cout<<"weights = "<<endl;
	print_vec(weights);
	H.set_params(weights);
	auto weights2 = H.get_params();
	cout<<"weights2 = "<<endl;
	print_vec(weights2);
}

void TrainingTesting(){
	int x_dim = 4;
	int y_dim = 3;
	int c_dim = 6;
	std::vector<int> gNeurs = {4,3};
	std::vector<int> fNeurs = {4,3};

	Hedging H(x_dim,y_dim,c_dim,"sigmoid","cos",gNeurs,fNeurs);
	
	std::vector<vec_vec> data(5);
	for (int i=0;i<5;i++){
		vec_vec path(10+i);
		for (int j = 0;j<10+i;j++){
			vec x = vec::Random(x_dim);
			path[j] = x;
		}

		auto last = path.back();
		vec v(1);
		v(0) = last.squaredNorm();

		path.push_back(v);

		data[i] = path;
	}

	
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
}

void PDETesting(){
	int in_dim = 2;
	int num_neurons = 10;
	std::string act_sym = "sigmoid";
	single_layer_nn N(in_dim,num_neurons,act_sym);

	vec lb(2);
	vec ub(2);
	lb<<0,0;
	ub<<1,1;

	int num_pts = 10000;

	std::shared_ptr<Lagrangian> pLC = std::make_shared<Laplace_Cube>(N, lb, ub, num_pts);
	CalcOfVar CV(pLC);
	double err = CV.error();
	cout<<err<<endl;

}