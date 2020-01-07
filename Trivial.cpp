#pragma once

#include<iostream>
#include <vector>
#include<memory>
#include<iterator>
#include<functional>

#include "_Extras/eigens/unsupported/Eigen/MatrixFunctions"
#include "Eigen/Dense"

typedef Eigen::MatrixXd mat;
typedef std::vector<mat> mat_vec;
typedef Eigen::VectorXd vec;
typedef std::vector<vec> vec_vec;

using namespace std;

class my_cube
{
public:
	std::vector<mat> elements;
	int n_rows;
	int n_cols;
	int n_slices;

	my_cube(){

	}
	my_cube(std::vector<mat>& v){
		this->elements = v;
		this->n_slices = this->elements.size();
		this->n_rows = this->elements[0].rows();
		this->n_cols = this->elements[0].cols();
	}
	my_cube(int slices,int rows, int cols){
		this->elements = std::vector<mat>(slices);
		this->n_slices = slices;
		this->n_rows = rows;
		this->n_cols = cols;
	}

	double& operator()(int k, int i,int j){
		if (k<this->n_slices and i<this->n_rows and j<this->n_cols){
			return this->elements[k](i,j);
		}
		else
		{
			std::cout<<"k = "<<k<<" slices = "<<this->n_slices;
			std::cout<<".... i = "<<i<<" rows = "<<this->n_rows;
			std::cout<<".... j = "<<j<<" cols = "<<this->n_cols<<std::endl;

			return this->elements[0](0,0);
		}
	}

	vec operator()(int i, int j){

		vec M(this->n_slices);
		for (int k=0;k<this->n_slices;k++){
			M(k) = (*this)(k,i,j);
		}

		return M;
	}
		
	mat get_row(int i){
		if (i>=this->n_rows)
		{
			std::cout<<"Index out of bounds"<<std::endl;
		}
		mat M(this->n_slices,this->n_cols);
		for (int k=0;k<this->n_slices;k++){
			for (int j=0;j<this->n_cols;j++){
				M(k,j) =(*this)(k,i,j); 
			}
		}
		return M;
	}

	void set_row(int i,mat M){
		if (i>=this->n_rows)
		{
			std::cout<<"Index out of bounds"<<std::endl;
		}
		for (int k=0;k<this->n_slices;k++){
			for (int j=0;j<this->n_cols;j++){
				(*this)(k,i,j) = M(k,j); 
			}
		}
	}

	mat get_col(int j){
		if (j>=this->n_cols)
		{
			std::cout<<"Index out of bounds"<<std::endl;
		}
		mat M(this->n_slices,this->n_rows);
		for (int k=0;k<this->n_slices;k++){
			for (int i=0;i<this->n_rows;i++){
				M(k,i) =(*this)(k,i,j); 
			}
		}
		return M;
	}

	void set_col(int j,mat M){
		if (j>=this->n_cols)
		{
			std::cout<<"Index out of bounds"<<std::endl;
		}
		
		for (int k=0;k<this->n_slices;k++){
			for (int i=0;i<this->n_rows;i++){
				(*this)(k,i,j) = M(k,i); 
			}
		}
	}

	mat& get_slice(int k){
		if (k>=this->n_slices)
		{
			std::cout<<"Index out of bounds"<<std::endl;
		}
		return (this->elements)[k];
	}

	void set_slice(int k,mat M){
		if (k>=this->n_slices)
		{
			std::cout<<"Index out of bounds"<<std::endl;
		}
		this->elements[k] = M;
	}

	void print(){
		int k = 0;
		std::cout<<std::endl;
		for (auto m:this->elements){
			std::cout<<"Slice "<<k<<std::endl;
			std::cout<<std::endl;
			std::cout<<m<<std::endl;
			k++;
		}
		std::cout<<std::endl;
		std::cout<<"THE END"<<std::endl;
		std::cout<<std::endl;
	}
};

