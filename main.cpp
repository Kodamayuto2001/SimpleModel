#define DATA_MAX	29000
#define HEIGHT		160
#define WIDTH		160
#define INPUT		HEIGHT*WIDTH	
#define HIDDEN		320
#define OUTPUT		29
#define MINI_BATCH_SIZE 100
//	#define PREDICT
#include "dataloader.hpp"
#include "FastSimpleNeuralNetwork.h"

int test1(void) {
	static dataset train_shuffle_path[DATA_MAX];
	static float x[INPUT];
	float t[OUTPUT];
	float loss;

	path_shuffle(
		"../SimpleNeuralNetwork2/29classes_dataset-main/train_data29/",
		train_shuffle_path
	);

	SimpleNeuralNetwork_init();

	for (int e = 0; e < 100; ++e) {
		for (int i = 0; i < DATA_MAX; ++i) {
			dataloader(train_shuffle_path[i].path, x);

			Flatten(&train_shuffle_path[i].label, t);

			SimpleNeuralNetwork(x, t, &loss);

			Adam();

			printf("\rlabel: %2d [%2d] [%5f]", train_shuffle_path[i].label, i, loss);

			if (i % 1000 == 0 && i != 0) { save("faceAI.model"); }
		}
	}
	return 0;
}

int test2(void) {
	static dataset train_shuffle_path[DATA_MAX];
	static float x[MINI_BATCH_SIZE][INPUT];
	static float t[MINI_BATCH_SIZE][OUTPUT];
	float loss = 0;

	path_shuffle(
		"../SimpleNeuralNetwork2/29classes_dataset-main/train_data29/",
		train_shuffle_path
	);

	SimpleNeuralNetwork_init();

	for (int e = 0; e < 1000; ++e) {
		for (int i = 0; i < DATA_MAX; ++i) {
			dataloader(
				train_shuffle_path[i].path,
				x[i % MINI_BATCH_SIZE]
			);

			Flatten(
				&train_shuffle_path[i].label, 
				t[i % MINI_BATCH_SIZE]
			);

			if (i % MINI_BATCH_SIZE == 0 && i != 0) {
				MiniBatchSimpleNeuralNetwork(x, t, &loss);

				Adam();
			}

			printf("\r [%2d] [%5f]", i, loss);
		}
	}
	return 0;
}

int main(void) {
	//test1();
	test2();
	return 0;
}