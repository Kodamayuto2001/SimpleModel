#ifndef _FastSimpleNeuralNetwork_HPP_
#define _FastSimpleNeuralNetwork_HPP_
#include<iostream>
#include<fstream>
#include<float.h>
#include<string>
#include<random>
#include<cmath>
using namespace std;

template<class ActFunc,class SoftmaxWithLoss>
class FastModel {
public:
    /*
	 *	FastMulÇÕïsïKóv
	 *	Forward
	 *	node		= x * weight(bias)
	 *
	 *	Backward
	 *	node		= dout * weight(bias)
	 *	weight(bias)= dout * x
	 */
	class Forward {
	public:
		double** weight;
		double* bias;
		double* node;	//	èÉì`î¿ãtì`î¿åìóp
	};
	class Backward {
	public:
		double** weight;
		double* bias;
	};
	Forward fc[2];
	Backward dfc[2];
	int size[3];
	double* y;
	
	FastModel(int&& inputSize, int&& hiddenSize, int&& outputSize) {
		size[0] = inputSize;
		size[1] = hiddenSize;
		size[2] = outputSize;
		

		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<double> dist(-1, 1);

		int i, j, k;
		for (i = 0; i < 2; ++i) {
			fc[i].weight = new double* [size[i + 1]];
			fc[i].bias = new double[size[i + 1]];
			fc[i].node = new double[size[i + 1]];

			dfc[i].weight = new double* [size[i + 1]];
			dfc[i].bias = new double[size[i + 1]];
			for (j = 0; j < size[i + 1]; ++j) {
				fc[i].weight[j] = new double[size[i]];
				dfc[i].weight[j] = new double[size[i]];

				fc[i].bias[j] = 0.0;
				fc[i].node[j] = 0.0;
				dfc[i].bias[j] = 0.0;
				for (k = 0; k < size[i]; ++k) {
					fc[i].weight[j][k] = dist(gen);
					dfc[i].weight[j][k] = 0.0;
				}
			}
		}

	}

	void del() {
		int i, j;
		for (i = 0; i < 2; ++i) {
			for (j = 0; j < size[i + 1]; ++j) {
				delete[] fc[i].weight[j];
				delete[] dfc[i].weight[j];
			}
			delete[] fc[i].weight;
			delete[] fc[i].bias;
			delete[] fc[i].node;
			delete[] dfc[i].weight;
			delete[] dfc[i].bias;
		}
	}
private:
	ActFunc* actf;
	SoftmaxWithLoss swl;
};

#endif // ! _FastSimpleNeuralNetwork_HPP_
