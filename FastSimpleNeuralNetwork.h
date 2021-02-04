#ifndef _FastSimpleNeuralNetwork_H_
#define _FastSimpleNeuralNetwork_H_
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef DATAMAX
#define DATAMAX		100
#endif // !DATAMAX
#ifndef CHANNEL
#define CHANNEL		1
#endif // !CHANNEL
#ifndef IMG_HEIGHT
#define IMG_HEIGHT	160
#endif // !IMG_HEIGHT
#ifndef IMG_WIDTH
#define IMG_WIDTH	160
#endif // !IMG_WIDTH
#ifndef INPUT_SIZE
#define INPUT_SIZE  CHANNEL*IMG_HEIGHT*IMG_WIDTH
#endif // !INPUT_SIZE
#ifndef HIDDEN_SIZE
#define HIDDEN_SIZE 320
#endif // !HIDDEN_SIZE
#ifndef OUTPUT_SIZE
#define OUTPUT_SIZE 100
#endif // !OUTPUT_SIZE

void Flatten(int, double[]);
void Sigmoid_forward(double*, double*);
void Sigmoid_backward(double*, double*, double*);
void ReLU_forward(double*, double*);
void ReLU_backward(double*, double*, double*);
void Softmax_forward(double*, double*);
void CrossEntropyError_forward(double*, double*, double*);
void SoftmaxWithLoss_backward(double*, double*, double*);
void SimpleNeuralNetwork_init();
void SimpleNeuralNetwork(
	void(*)(double*, double*),
	void(*)(double*, double*),
	void(*)(double*, double*, double*),
	void(*)(double*, double*, double*),
	void(*)(double*, double*, double*),
	double*,
	double*,
	double*
);

//	�O���[�o���ϐ��@�X�^�e�B�b�N�̈�@
//  �v���O�����J�n����I���܂Ń��������蓖�ĕω����Ȃ�
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
		node_1[i] = 0.0;
		dbias_1[i] = 0.0;
		dnode_1[i] = 0.0;
		for (int j = 0; j < INPUT_SIZE; ++j) {
			weight_1[i][j] = 1 - ((double)rand() / (RAND_MAX / 2));
			dweight_1[i][j] = 0.0;
		}
	}
	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		bias_2[i] = 0.0;
		node_2[i] = 0.0;
		dbias_2[i] = 0.0;
		for (int j = 0; j < HIDDEN_SIZE; ++j) {
			weight_2[i][j] = 1 - ((double)rand() / (RAND_MAX / 2));
			dweight_2[i][j] = 0.0;
		}
	}
}

void SimpleNeuralNetwork(
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
		dnode_1[i] = 0.0;
		for (j = 0; j < INPUT_SIZE; ++j) {
			dweight_1[i][j] = x[j] * dbias_1[i];
		}
	}
}
void Flatten(int label, double x[]) {
	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		x[i] = 0;
	}
	x[label] = 1;
}
void Sigmoid_forward(double* x, double* y) {
	*y = 1 / (1.0 + exp(-(*x)));
}
void Sigmoid_backward(double* y, double* dout, double* dx) {
	*dx = (*dout) * (1.0 - (*y)) * (*y);
}
void ReLU_forward(double* x, double* y) {
	if ((*x) > 0.0) { (*y) = (*x); }
	else { (*y) = 0.0; }
}
void ReLU_backward(double* y, double* dout, double* dx) {
	if ((*y) > 0.0) { (*dx) = (*dout); }
	else { (*dx) = 0.0; }
}
void Softmax_forward(double* x, double* y) {
	double a = x[0];
	double tmp[OUTPUT_SIZE];
	double b = 0.0;
	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		if (a < x[i]) { a = x[i]; }
	}
	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		tmp[i] = exp(x[i] - a);
		b += tmp[i];
	}
	a = 1 / b;
	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		y[i] = tmp[i] * a;
	}
}
void CrossEntropyError_forward(double* t, double* y, double* loss) {
	double a = 0.0;
	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		a += log(y[i]+ 1.0e-300) * t[i];
	}
	*loss = a * (-1);
}
void SoftmaxWithLoss_backward(double* t, double* y, double* dx) {
	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		dx[i] = y[i] - t[i];
	}
}

void SGD(double lr = 0.01) {
	for (int i = 0; i < HIDDEN_SIZE; ++i) {
		bias_1[i] -= lr * dbias_1[i];
		for (int j = 0; j < INPUT_SIZE; ++j) {
			weight_1[i][j] -= lr * dweight_1[i][j];
		}
	}
	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		bias_2[i] -= lr * dbias_2[i];
		for (int j = 0; j < HIDDEN_SIZE; ++j) {
			weight_2[i][j] -= lr * dweight_2[i][j];
		}
	}
}


void Momentum(double lr = 0.01, double momentum = 0.9) {
	static int flag = 0;
	static double v_weight_1[HIDDEN_SIZE][INPUT_SIZE];
	static double v_weight_2[OUTPUT_SIZE][HIDDEN_SIZE];
	static double v_bias1[HIDDEN_SIZE];
	static double v_bias2[OUTPUT_SIZE];
	if (flag == 0) {
		for (int i = 0; i < HIDDEN_SIZE; ++i) {
			v_bias1[i] = 0;
			for (int j = 0; j < INPUT_SIZE; ++j) {
				v_weight_1[i][j] = 0;
			}
		}
		for (int i = 0; i < OUTPUT_SIZE; ++i) {
			v_bias2[i] = 0;
			for (int j = 0; j < HIDDEN_SIZE; ++j) {
				v_weight_2[i][j] = 0;
			}
		}
	}
	for (int i = 0; i < HIDDEN_SIZE; ++i) {
		v_bias1[i] = momentum * v_bias1[i] - lr * dbias_1[i];
		bias_1[i] += v_bias1[i];
		for (int j = 0; j < INPUT_SIZE; ++j) {
			v_weight_1[i][j] = momentum * v_weight_1[i][j] - lr * dweight_1[i][j];
			weight_1[i][j] += v_weight_1[i][j];
		}
	}
	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		v_bias2[i] = momentum * v_bias2[i] - lr * dbias_2[i];
		bias_2[i] += v_bias2[i];
		for (int j = 0; j < HIDDEN_SIZE; ++j) {
			v_weight_2[i][j] = momentum * v_weight_2[i][j] - lr * dweight_2[i][j];
			weight_2[i][j] += v_weight_2[i][j];
		}
	}

	flag = 1;
}

void AdaGrad(double lr = 0.01) {
	static int flag = 0;
	static double h_weight_1[HIDDEN_SIZE][INPUT_SIZE];
	static double h_weight_2[OUTPUT_SIZE][HIDDEN_SIZE];
	static double h_bias_1[HIDDEN_SIZE];
	static double h_bias_2[OUTPUT_SIZE];
	static constexpr double Delta = 1.0e-300;
	if (flag == 0) {
		for (int i = 0; i < HIDDEN_SIZE; ++i) {
			h_bias_1[i] = 0;
			for (int j = 0; j < INPUT_SIZE; ++j) {
				h_weight_1[i][j] = 0;
			}
		}
		for (int i = 0; i < OUTPUT_SIZE; ++i) {
			h_bias_2[i] = 0;
			for (int j = 0; j < HIDDEN_SIZE; ++j) {
				h_weight_2[i][j] = 0;
			}
		}
	}
	for (int i = 0; i < HIDDEN_SIZE; ++i) {
		h_bias_1[i] += dbias_1[i] * dbias_1[i];
		bias_1[i] -= lr / sqrt(h_bias_1[i] + Delta) * dbias_1[i];
		for (int j = 0; j < INPUT_SIZE; ++j) {
			h_weight_1[i][j] += dweight_1[i][j] * dweight_1[i][j];
			weight_1[i][j] -= lr / sqrt(h_weight_1[i][j] + Delta) * dweight_1[i][j];
		}
	}
	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		h_bias_2[i] += dbias_2[i] * dbias_2[i];
		bias_2[i] -= lr / sqrt(h_bias_2[i] + Delta) * dbias_2[i];
		for (int j = 0; j < HIDDEN_SIZE; ++j) {
			h_weight_2[i][j] += dweight_2[i][j] * dweight_2[i][j];
			weight_2[i][j] -= lr / sqrt(h_weight_2[i][j] + Delta) * dweight_2[i][j];
		}
	}
}

#endif // !_FastSimpleNeuralNetwork_H_