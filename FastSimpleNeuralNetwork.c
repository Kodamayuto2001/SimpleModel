#include "FastSimpleNeuralNetwork.h"

int main(void) {
	double x[INPUT_SIZE] = { 5,1 };
	double t[INPUT_SIZE] = { 1,0 };
	double loss;

	SimpleNeuralNetwork_init();
	SimpleNeuralNetwork(
		ReLU_forward,
		Softmax_forward,
		CrossEntropyError_forward,
		SoftmaxWithLoss_backward,
		ReLU_backward,
		x, t, &loss
	);
	printf("%f\n", loss);

	return 0;
}