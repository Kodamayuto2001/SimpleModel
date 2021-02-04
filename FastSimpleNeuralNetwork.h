#ifndef _FastSimpleNeuralNetwork_H_
#define _FastSimpleNeuralNetwork_H_
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DATAMAX		100
#define CHANNEL		1
#define IMG_HEIGHT	2
#define IMG_WIDTH	1

#define INPUT_SIZE  CHANNEL*IMG_HEIGHT*IMG_WIDTH
#define HIDDEN_SIZE 3
#define OUTPUT_SIZE 2

int* Flatten(int, int[OUTPUT_SIZE]);
void Sigmoid_forward(double*, double*);
void Sigmoid_backward(double*, double*, double*);
void ReLU_forward(double*, double*);
void ReLU_backward(double*, double*, double*);
void Softmax_forward(double*, double*);
void CrossEntropyError_forward(double*, double*, double*);
void SoftmaxWithLoss_backward(double*, double*, double*);
void SimpleNeuralNetwork_init();
void* SimpleNeuralNetwork(
	void(*)(double*, double*),
	void(*)(double*, double*),
	void(*)(double*, double*, double*),
	void(*)(double*, double*, double*),
	void(*)(double*, double*, double*),
	double*,
	double*,
	double*
);

//	グローバル変数　スタティック領域　
//  プログラム開始から終わりまでメモリ割り当て変化しない
double weight_1[HIDDEN_SIZE][INPUT_SIZE];
double weight_2[OUTPUT_SIZE][HIDDEN_SIZE];
double bias_1[HIDDEN_SIZE];
double bias_2[OUTPUT_SIZE];
double node_1[HIDDEN_SIZE];
double node_2[OUTPUT_SIZE];
double dweight_1[HIDDEN_SIZE][INPUT_SIZE];
double dweight_2[OUTPUT_SIZE][HIDDEN_SIZE];
double dbias_1[HIDDEN_SIZE];
double dbias_2[OUTPUT_SIZE];
double dnode_1[HIDDEN_SIZE];


void SimpleNeuralNetwork_init() {
	srand(time(NULL));
	for (int i = 0; i < HIDDEN_SIZE; ++i) {
		bias_1[i] = 0.0;
		for (int j = 0; j < INPUT_SIZE; ++j) {
			weight_1[i][j] = 1 - ((double)rand() / (RAND_MAX / 2));
		}
	}
	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		bias_2[i] = 0.0;
		for (int j = 0; j < HIDDEN_SIZE; ++j) {
			weight_2[i][j] = 1 - ((double)rand() / (RAND_MAX / 2));
		}
	}
}

void* SimpleNeuralNetwork(
	void(*forward_1)(double*, double*),
	void(*forward_2)(double*, double*),
	void(*lossFunc)(double*, double*, double*),
	void(*backward_2)(double*, double*, double*),
	void(*backward_1)(double*, double*, double*),
	double* x,
	double* t,
	double* loss
) {
	int i, j;
	for (i = 0; i < HIDDEN_SIZE; ++i) {
		dnode_1[i] = 0.0;
		for (j = 0; j < INPUT_SIZE; ++j) {
			node_1[i] += x[j] * weight_1[i][j];
		}
		node_1[i] += bias_1[i];
		forward_1(&node_1[i], &node_1[i]);
	}
	for (i = 0; i < OUTPUT_SIZE; ++i) {
		for (j = 0; j < HIDDEN_SIZE; ++j) {
			node_2[i] += node_1[j] * weight_2[i][j];
		}
		node_2[i] += bias_2[i];
	}
	forward_2(node_2, node_2);
	lossFunc(t, node_2, loss);
	backward_2(t, node_2, dbias_2);
	for (i = 0; i < OUTPUT_SIZE; ++i) {
		node_2[i] = 0.0;
		for (j = 0; j < HIDDEN_SIZE; ++j) {
			dnode_1[j] += dbias_2[i] * weight_2[i][j];
			dweight_2[i][j] = node_1[j] * dbias_2[i];
		}
	}
	for (i = 0; i < HIDDEN_SIZE; ++i) {
		backward_1(&node_1[i], &dnode_1[i], &dbias_1[i]);
		node_1[i] = 0.0;
		for (j = 0; j < INPUT_SIZE; ++j) {
			dweight_1[i][j] = x[j] * dbias_1[i];
		}
	}
}
int* Flatten(int label, int x[OUTPUT_SIZE]) {
	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		x[i] = 0;
	}
	x[label] = 1;
	return x;
}
void Sigmoid_forward(double* x, double* y) {
	*y = 1 / (1.0 + exp(-(*x)));
}
void Sigmoid_backward(double* y, double* dout, double* dx) {
	*dx = (*dout) * (1.0 - (*y)) * (*y);
}
void ReLU_forward(double* x, double* y) {
	if (*x > 0.0) { *y = *x; }
	else { *y = 0.0; }
}
void ReLU_backward(double* y, double* dout, double* dx) {
	if (*y > 0.0) { *dx = *dout; }
	else { *dx = 0.0; }
}
void Softmax_forward(double* x, double* y) {
	double a = x[0];
	double tmp[OUTPUT_SIZE];

	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		if (a < x[i]) { a = x[i]; }
	}
	{
		double b = 0.0;
		for (int i = 0; i < OUTPUT_SIZE; ++i) {
			tmp[i] = exp(x[i] - a);
			b += tmp[i];
		}
		a = 1 / b;
	}
	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		y[i] = tmp[i] * a;
	}
}
void CrossEntropyError_forward(double* t, double* y, double* loss) {
	double a = 0.0;
	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		a += log(y[i]) * t[i];
	}
	*loss = a * (-1);
}
void SoftmaxWithLoss_backward(double* t, double* y, double* dx) {
	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		dx[i] = y[i] - t[i];
	}
}
#endif // !_FastSimpleNeuralNetwork_H_
