#include "dataset.hpp"
#include "dataloader.hpp"
#define DATAMAX		100
#define CHANNEL		1
#define IMG_HEIGHT	160
#define IMG_WIDTH	160
#define INPUT_SIZE  CHANNEL*IMG_HEIGHT*IMG_WIDTH
#define HIDDEN_SIZE 320
#define OUTPUT_SIZE 100
#include "FastSimpleNeuralNetwork.h"


int main(void) {
	DataLoader dl("DataSet/", DATAMAX, CHANNEL, IMG_HEIGHT, IMG_WIDTH);
	dl.load();
	double** x = dl.vecImg();
	double t[OUTPUT_SIZE];
	Flatten(0, t);

	double loss;
	SimpleNeuralNetwork_init();
	for (int e = 0; e < 1; ++e) {
		for (int i = 0; i < DATAMAX; ++i) {
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
			AdaGrad();
			cout << loss << endl;
		}
	}
	save("test.model");
	dl.del();

	load("test.model");
	return 0;
}