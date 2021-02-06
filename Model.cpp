#include "dataset.hpp"
#define DATAMAX		100
#define CHANNEL		1
#define IMG_HEIGHT	160
#define IMG_WIDTH	160
#define INPUT_SIZE  CHANNEL*IMG_HEIGHT*IMG_WIDTH
#define HIDDEN_SIZE 320
#define OUTPUT_SIZE 10
#include "FastSimpleNeuralNetwork.h"
#include "dataloader.hpp"

double x[DATAMAX][INPUT_SIZE];

int main(void) {
	dataloader("DataSet/", DATAMAX, CHANNEL, IMG_HEIGHT, IMG_WIDTH, x);
	double t[OUTPUT_SIZE];
	Flatten(0, t);

	double loss;
	SimpleNeuralNetwork_init();
	for (int e = 0; e < 100; ++e) {
		for (int i = 0; i < DATAMAX-1; ++i) {
			SimpleNeuralNetwork(
				Sigmoid_forward,
				Softmax_forward,
				CrossEntropyError_forward,
				SoftmaxWithLoss_backward,
				Sigmoid_backward,
				x[i],
				t,
				&loss
			);
			Adam();
			// cout << loss << endl;
			cout << i << endl;
		}
	}
	save("test.model");

	return 0;
}