#include "dataset.hpp"
#include "dataloader.hpp"
#include "FastSimpleNeuralNetwork.hpp"

void MakeDataFunc() {
	DataSet ds;
	ds.MakeDataSet(100);
}

void train_0() {
	typedef FastModel<FastSigmoid, FastSoftmaxWithLoss> Net;
	Net model(2, 3, 2);
	double x[2] = { 2.35,4.97 };
	double t[2] = { 1,0 };
	FastAdam<Net> optimizer;
	for (int e = 0; e < 10000; ++e) {
		model.forward(x, t);
		model.backward();
		optimizer.step(&model);
		cout << model.loss << endl;
	}
	model.save("model.kodamayuto");

	model.load("model.kodamayuto");
	double z[2] = { 2.5,5.0 };
	model.predict(z);
	cout << model.node[1][0]*100 << "%" << endl;

	model.del();
}

void train_1() {
	int dataMax = 100;
	int channel = 1;
	int imgHeight = 160;
	int imgWidth = 160;
	double lr = 0.00001;
	double beta1 = 0.9;
	double beta2 = 0.999;
	int hiddenNeuron = 320;
	int outSize = 10;

	typedef FastModel<FastReLU, FastSoftmaxWithLoss> Net;
	DataLoader dl("DataSet/", dataMax,channel,imgHeight,imgWidth);
	dl.load();
	double** x = dl.vecImg();
	double t[10] = { 0,0,0,0,0,0,0,1,0,0 };

	Net model(
		move(imgHeight*imgWidth*channel), 
		move(hiddenNeuron), 
		move(outSize)
	);
	FastAdam<Net> optimizer(lr,beta1,beta2);

	for (int e = 0; e < 1; ++e) {
		for (int i = 0; i < 80; ++i) {
			model.forward(x[i], t);
			model.backward();
			optimizer.step(&model);
			cout << model.loss << endl;
		}
	}
	model.save("model.kodamayuto");

	model.load("model.kodamayuto");
	model.predict(x[96]);
	cout << model.node[1][7] * 100 << "%" << endl;
	model.del();
	dl.del();
}

int main() {
	train_1();
	return 0;
}