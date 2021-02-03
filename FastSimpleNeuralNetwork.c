#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DATAMAX		100
#define CHANNEL		1
#define IMG_HEIGHT	1
#define IMG_WIDTH	2

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
void SimpleNeuralNetwork(
	void(*)(double*,double*),
	void(*)(double*,double*),
	void(*)(double*,double*,double*),
	void(*)(double*,double*,double*),
	void(*)(double*,double*,double*),
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

int main(void) {
	double x[INPUT_SIZE] = { 5,1 };
	double t[INPUT_SIZE] = { 1,0 };

	SimpleNeuralNetwork_init();
	SimpleNeuralNetwork(
		ReLU_forward,
		Softmax_forward,
		CrossEntropyError_forward,
		SoftmaxWithLoss_backward,
		ReLU_backward,
		x, t
	);
	
}

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

void SimpleNeuralNetwork(
	void(*forward_1)(double*,double*),
	void(*forward_2)(double*,double*),
	void(*loss)(double*,double*,double*),
	void(*backward_2)(double*,double*,double*),
	void(*backward_1)(double*,double*,double*),
	double* x,
	double* t
) {
	double a;
	for (int i = 0; i < HIDDEN_SIZE; ++i) {
		a = 0.0;
		for (int j = 0; j < INPUT_SIZE; ++j) {
			a += x[j] * weight_1[i][j];
			dweight_1[i][j] = x[j];
		}
		a += bias_1[i];
		forward_1(&a, &node_1[i]);
	}

	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		a = 0.0;
		for (int j = 0; j < HIDDEN_SIZE; ++j) {
			a += node_1[j] * weight_2[i][j];
		}
	}
	return ;
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
void Sigmoid_backward(double* dout, double* dx, double* y) {
	*dx = (*dout) * (1.0 - (*y)) * (*y);
}
void ReLU_forward(double* x, double* y) {
	if (*x > 0.0) { *y = *x; }
	else { *y = 0.0; }
}
void ReLU_backward(double* dout, double* dx, double* x) {
	if (*x > 0.0) { *dx = *dout; }
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
void CrossEntropyError_forward(double* t, double* loss, double* y) {
	double a = 0.0;
	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		a += log(y[i]) * t[i];
	}
	*loss = a * (-1);
}
void SoftmaxWithLoss_backward(double* dx, double* y, double* t) {
	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		dx[i] = y[i] - t[i];
	}
}