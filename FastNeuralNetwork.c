#include <stdio.h>
#include <math.h>
#define DATAMAX		100
#define CHANNEL		1
#define IMG_HEIGHT	160
#define IMG_WIDTH	160

#define INPUT_SIZE  CHANNEL*IMG_HEIGHT*IMG_WIDTH
#define HIDDEN_SIZE 320
#define OUTPUT_SIZE 10

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
void ReLU_backward(double* dout, double* dx,double* x) {
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

void SimpleNeuralNetwork(
	void (*forward_1)(double*,double*),
	void (*forward_2)(double*,double*),
	void (*loss)(double*,double*,double*),
	void (*backward_2)(double*,double*,double*),
	void (*backward_1)(double*,double*,double*)
) {
	
}

#ifdef TEST
void test_0(void (*pf)(double*, double*)) {
	double x = -1;
	double y;
	pf(&x,&y);
	printf("%lf\n", y);
}
void test_1() {
	int x[OUTPUT_SIZE];
	int* a = Flatten(2, x);
	printf("%d %d %d\n", a[0], a[1], a[2]);
}
#endif

int main(void) {
	
}
